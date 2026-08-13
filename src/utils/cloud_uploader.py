"""
Cloud Uploader Module for ECG Reports
Supports multiple cloud storage services for automatic report backup
Includes offline queue support for automatic sync when internet is restored
"""

import os
import re
import json
import requests

from pathlib import Path
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
import sys

# Load environment variables
# 1) Load from current working directory (if running from project root)
load_dotenv()
# 2) Also attempt to load from project root explicitly (works when app starts from elsewhere)
try:
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[2] if len(this_file.parents) >= 3 else this_file.parent
    root_env = project_root / '.env'
    if root_env.exists():
        load_dotenv(dotenv_path=str(root_env), override=False)
    else:
        # Fallback: also try one directory up (in case of unusual packaging)
        alt_env = project_root.parent / '.env'
        if alt_env.exists():
            load_dotenv(dotenv_path=str(alt_env), override=False)
except Exception:
    # Best-effort; silently continue if path resolution fails
    pass

# 3) PyInstaller: load .env from frozen app directory and unpack dir (sys._MEIPASS)
try:
    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(sys.executable)
        app_env = os.path.join(app_dir, '.env')
        if os.path.exists(app_env):
            load_dotenv(dotenv_path=app_env, override=False)
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            meipass_env = os.path.join(meipass, '.env')
            if os.path.exists(meipass_env):
                load_dotenv(dotenv_path=meipass_env, override=False)
except Exception:
    pass

# Final fallback: manually parse .env if python-dotenv misses it (rare edge cases)
def _manual_env_load(env_path: Path):
    try:
        if env_path.exists():
            with env_path.open('r', encoding='utf-8', errors='ignore') as f:
                for raw in f:
                    line = raw.replace('\ufeff', '')  # strip BOM if present
                    line = line.replace('＝', '=')      # normalize unicode equals
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        k, v = line.split('=', 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        # Force override to ensure latest values are used
                        if k:
                            os.environ[k] = v
    except Exception:
        pass

try:
    _manual_env_load(root_env)
    # Also try app dir and MEIPASS for manual parse
    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(sys.executable)
        _manual_env_load(Path(app_dir) / '.env')
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            _manual_env_load(Path(meipass) / '.env')
except Exception:
    pass


# Report files are named  <DEVICE>_<YYYYMMDD>_<HHMMSS>.<ext>  — e.g.
# "ECG_Report_12_1_DM ECG V1.0 A010_20260810_154532.pdf" → ("A010", "20260810",
# "154532"). This triple is the report's identity everywhere in the pipeline:
# the doctor-review Lambda parses it out of the filename it is given and looks
# for the report under that device and date. Anything filed elsewhere is
# invisible to it and comes back as HTTP 404 "Old Report not supported."
_REPORT_IDENTITY_RE = re.compile(r'([A-Za-z0-9]{2,20})_(\d{8})_(\d{6})\.[A-Za-z0-9]+$')

# Reports whose name carries the timestamp but no device token
# ("ECG_Report_12_1_20260810_154532.pdf"). The backend accepts these when the
# device id is supplied separately, so the date is still usable for filing.
_REPORT_TIMESTAMP_RE = re.compile(r'_(\d{8})_(\d{6})\.[A-Za-z0-9]+$')


def _report_identity(filename: str):
    """
    Return (device_id, 'YYYYMMDD', 'HHMMSS') parsed from a report filename.

    Returns None when the name carries no device token, in which case callers
    fall back to the device this machine is registered to.
    """
    match = _REPORT_IDENTITY_RE.search(str(filename or "").strip())
    if not match:
        return None
    return match.group(1).upper(), match.group(2), match.group(3)


def _report_timestamp(filename: str):
    """Return ('YYYYMMDD', 'HHMMSS') from a report filename, or None."""
    match = _REPORT_TIMESTAMP_RE.search(str(filename or "").strip())
    if not match:
        return None
    return match.group(1), match.group(2)


def _identity_date_path(date_str: str) -> str:
    """'20260810' → '2026/08/10'."""
    return f"{date_str[0:4]}/{date_str[4:6]}/{date_str[6:8]}"


class CloudUploader:
    """Handle uploading ECG reports to cloud storage with offline queue support"""

    def __init__(self):
        self.cloud_service = os.getenv('CLOUD_SERVICE', 'none').lower()
        self.upload_enabled = os.getenv('CLOUD_UPLOAD_ENABLED', 'false').lower() == 'true'
        
        # AWS S3 Configuration
        self.s3_bucket = os.getenv('AWS_S3_BUCKET')
        self.s3_region = os.getenv('AWS_S3_REGION', 'us-east-1')
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        # Azure Blob Storage Configuration
        self.azure_connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        self.azure_container = os.getenv('AZURE_CONTAINER_NAME', 'ecg-reports')
        
        # Google Cloud Storage Configuration
        self.gcs_bucket = os.getenv('GCS_BUCKET_NAME')
        self.gcs_credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        # Custom API Endpoint Configuration
        self.api_endpoint = os.getenv('CLOUD_API_ENDPOINT')
        self.api_key = os.getenv('CLOUD_API_KEY')
        
        # FTP/SFTP Configuration
        self.ftp_host = os.getenv('FTP_HOST')
        self.ftp_port = int(os.getenv('FTP_PORT', '21'))
        self.ftp_username = os.getenv('FTP_USERNAME')
        self.ftp_password = os.getenv('FTP_PASSWORD')
        self.ftp_remote_path = os.getenv('FTP_REMOTE_PATH', '/ecg-reports')
        
        # Dropbox Configuration
        self.dropbox_token = os.getenv('DROPBOX_ACCESS_TOKEN')
        
        # Doctor Review API Configuration
        self.doctor_review_enabled = os.getenv('DOCTOR_REVIEW_ENABLED', 'false').lower() == 'true'
        self.doctor_review_api_url = (
            os.getenv('DOCTOR_UPLOAD_API_URL')
            or os.getenv('DOCTOR_REVIEW_API_URL', '')
        ).strip()
        self.doctor_review_api_key = (
            os.getenv('DOCTOR_UPLOAD_API_KEY')
            or os.getenv('DOCTOR_REVIEW_API_KEY', '')
        ).strip()
        self.reviewed_reports_api_url = os.getenv(
            'REVIEWED_REPORTS_API_URL',
            'https://6jhix49qt6.execute-api.us-east-1.amazonaws.com/api/public/reviewed-reports'
        ).strip()
        self.reviewed_reports_api_key = (
            os.getenv('PUBLIC_API_KEY')
            or os.getenv('REVIEWED_REPORTS_API_KEY', '')
        ).strip()
        self._doctor_list_cache = None
        self._last_doctor_fetch_time = 0
        
        # Log file for upload tracking
        self.upload_log_path = "reports/upload_log.json"
        
        # Initialize offline queue for automatic sync
        try:
            from .offline_queue import get_offline_queue
            self.offline_queue = get_offline_queue()
            print("[OK] Offline queue initialized for cloud uploader")
        except Exception as e:
            print(f"[WARN] Could not initialize offline queue: {e}")
            self.offline_queue = None

    def reload_config(self):
        """Re-read .env from CWD and project root and refresh fields."""
        try:
            load_dotenv(override=True)
            this_file = Path(__file__).resolve()
            project_root = this_file.parents[2] if len(this_file.parents) >= 3 else this_file.parent
            root_env = project_root / '.env'
            if root_env.exists():
                load_dotenv(dotenv_path=str(root_env), override=True)
            # Manual fallback parse as well
            try:
                _manual_env_load(root_env)
            except Exception:
                pass
            # Also try CWD .env
            try:
                cwd_env = Path(os.getcwd()) / '.env'
                _manual_env_load(cwd_env)
            except Exception:
                pass
        except Exception:
            pass
        self.cloud_service = os.getenv('CLOUD_SERVICE', 'none').lower()
        self.upload_enabled = os.getenv('CLOUD_UPLOAD_ENABLED', 'false').lower() == 'true'
        self.s3_bucket = os.getenv('AWS_S3_BUCKET')
        self.s3_region = os.getenv('AWS_S3_REGION', 'us-east-1')
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.doctor_review_enabled = os.getenv('DOCTOR_REVIEW_ENABLED', 'false').lower() == 'true'
        self.doctor_review_api_url = (
            os.getenv('DOCTOR_UPLOAD_API_URL')
            or os.getenv('DOCTOR_REVIEW_API_URL', '')
        ).strip()
        self.doctor_review_api_key = (
            os.getenv('DOCTOR_UPLOAD_API_KEY')
            or os.getenv('DOCTOR_REVIEW_API_KEY', '')
        ).strip()
        self.reviewed_reports_api_url = os.getenv(
            'REVIEWED_REPORTS_API_URL',
            'https://6jhix49qt6.execute-api.us-east-1.amazonaws.com/api/public/reviewed-reports'
        ).strip()
        self.reviewed_reports_api_key = (
            os.getenv('PUBLIC_API_KEY')
            or os.getenv('REVIEWED_REPORTS_API_KEY', '')
        ).strip()

    def get_config_snapshot(self):
        return {
            'cloud_service': self.cloud_service,
            'upload_enabled': self.upload_enabled,
            's3_bucket': self.s3_bucket,
            's3_region': self.s3_region,
            'aws_access_key_set': bool(self.aws_access_key),
            'aws_secret_key_set': bool(self.aws_secret_key),
        }
        
    def is_configured(self):
        """Check if cloud upload is properly configured"""
        if not self.upload_enabled:
            return False
            
        if self.cloud_service == 's3':
            return bool(self.s3_bucket and self.aws_access_key and self.aws_secret_key)
        elif self.cloud_service == 'azure':
            return bool(self.azure_connection_string)
        elif self.cloud_service == 'gcs':
            return bool(self.gcs_bucket and self.gcs_credentials_path)
        elif self.cloud_service == 'api':
            return bool(self.api_endpoint)
        elif self.cloud_service == 'ftp' or self.cloud_service == 'sftp':
            return bool(self.ftp_host and self.ftp_username)
        elif self.cloud_service == 'dropbox':
            return bool(self.dropbox_token)
        
        return False

    @staticmethod
    def _safe_int_env(name: str, default: int) -> int:
        try:
            return int(float(os.getenv(name, str(default))))
        except Exception:
            return int(default)

    @staticmethod
    def _safe_float_env(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return float(default)

    def _extract_mobile_for_s3_key(self, metadata: Optional[dict]) -> str:
        if not isinstance(metadata, dict):
            return ""
        candidates = [
            metadata.get("master_phone"),
            metadata.get("login_identifier"),
            metadata.get("mobile_no"),
            metadata.get("mobile_number"),
            metadata.get("phone"),
            metadata.get("patient_phone"),
            metadata.get("user_id"),
        ]
        for val in candidates:
            digits = "".join(ch for ch in str(val or "") if ch.isdigit())
            if len(digits) >= 10:
                return digits[-10:]
        return ""

    def _build_s3_key(self, filename: str, metadata: Optional[dict]) -> str:
        meta = metadata or {}
        report_type = str(meta.get("report_type") or "").strip().lower()
        report_type = report_type.replace(" ", "_").replace("-", "_")
        allowed_report_types = {"12_lead_ecg", "hrv", "hyperkalemia"}
        if report_type not in allowed_report_types:
            report_type = ""

        # NO automatic prefixing for JSON files. Keep filenames matching the PDF
        # for easier identification in the S3 browser.
        key_filename = filename

        prefix = (os.getenv("S3_REPORTS_PREFIX", "reports") or "reports").strip().strip("/")

        # The report's own identity decides where it is filed — NOT the clock or
        # whichever device happens to be plugged in right now. A report recorded
        # on the 10th but uploaded on the 12th used to land under 2026/08/12, and
        # one uploaded with the RhythmUltra unplugged landed under "0000" even
        # though its filename says A010. Either mismatch makes the review Lambda
        # (which matches on device id + the <DEVICE>_<DATE>_<TIME>.pdf key suffix)
        # answer "Old Report not supported."
        machine_serial_dir, date_path = self._report_location(filename)

        mobile = self._extract_mobile_for_s3_key(meta)
        if mobile:
            return f"{prefix}/{mobile}/{machine_serial_dir}/{key_filename}"
        return f"{prefix}/{date_path}/{machine_serial_dir}/{key_filename}"

    def _report_location(self, filename: str) -> tuple:
        """
        Return (device_dir, 'YYYY/MM/DD') for a report file.

        Both come from the report's own name where possible; the device falls
        back to this machine's RhythmUltra serial (connected, else registered)
        and the date to today only when the name carries neither.
        """
        identity = _report_identity(filename)
        if identity:
            device, date_str, _ = identity
            # For a PDF the filename token IS the device the backend will look
            # the report up under, so it always wins. Sidecars (JSON payloads,
            # HL7) are never looked up that way and some are named after the
            # user's phone number — "12_lead_9560350477_20260810_175623.json"
            # would otherwise create a folder named after the phone — so those
            # keep this machine's device folder unless the token looks like a
            # device tag (short, not all digits, e.g. A010).
            is_pdf = str(filename).lower().endswith(".pdf")
            if is_pdf or (len(device) <= 6 and not device.isdigit()):
                return device, _identity_date_path(date_str)

        stamp = _report_timestamp(filename)
        date_path = _identity_date_path(stamp[0]) if stamp else datetime.now().strftime("%Y/%m/%d")

        machine_serial = ""
        try:
            from utils.license_manager import get_rhythmultra_serial
            machine_serial = get_rhythmultra_serial() or ""
        except Exception:
            machine_serial = ""
        if not machine_serial:
            try:
                from utils.license_manager import load_registration_profile
                profile = load_registration_profile()
                machine_serial = (profile.get("rhythmultra_serial")
                                  or profile.get("rhythmulta_serial")
                                  or profile.get("RhythmUltra_serial")
                                  or "")
            except Exception:
                machine_serial = ""

        machine_serial_clean = ''.join(c for c in machine_serial if c.isalnum())
        device_dir = machine_serial_clean[-4:].upper() if len(machine_serial_clean) >= 4 else "0000"
        return device_dir, date_path
    
    def _is_file_already_uploaded(self, file_path):
        """
        Check if a file has already been uploaded based on filename
        
        Args:
            file_path (str): Path to the file to check
            
        Returns:
            bool: True if file has already been uploaded, False otherwise
        """
        try:
            filename = os.path.basename(file_path)
            
            # Check upload log
            if os.path.exists(self.upload_log_path):
                try:
                    with open(self.upload_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        raw = f.read().strip()
                    log_data = json.loads(raw) if raw else []
                except Exception:
                    log_data = []
                
                # Check if this filename has been uploaded before
                for entry in log_data:
                    # Get filename from the logged path
                    logged_filename = os.path.basename(entry.get('local_path', ''))
                    
                    # Also check metadata filename as backup
                    metadata_filename = entry.get('metadata', {}).get('filename', '')
                    
                    # If filename matches and upload was successful
                    if (logged_filename == filename or metadata_filename == filename):
                        if entry.get('result', {}).get('status') == 'success':
                            return True
            
            return False
            
        except Exception as e:
            print(f"[WARN] Error checking upload history: {e}")
            # If we can't check, assume not uploaded (safer to upload twice than not at all)
            return False
    
    def upload_report(self, file_path, metadata=None):
        """
        Upload ONLY reports, metrics, and report files to AWS S3
        Does NOT upload: session logs, debug data, crash logs, temp files
        Prevents duplicate uploads - files are only uploaded once
        Supports offline mode - queues for upload when internet is restored
        
        Args:
            file_path (str): Path to the report file (PDF, JSON, etc.)
            metadata (dict): Optional metadata about the report
            
        Returns:
            dict: Upload result with status, url, and error if any
        """
        if not self.upload_enabled:
            return {"status": "disabled", "message": "Cloud upload is disabled"}
            
        if not self.is_configured():
            return {"status": "error", "message": f"Cloud service '{self.cloud_service}' is not properly configured"}
        
        try:
            # Check if file has already been uploaded
            filename = os.path.basename(file_path)
            if self._is_file_already_uploaded(file_path):
                return {
                    "status": "already_uploaded",
                    "message": f"File '{filename}' has already been uploaded to cloud - skipping duplicate upload",
                    "filename": filename
                }
            
            # Check if this is an uploadable report/metric payload
            file_ext = Path(file_path).suffix.lower()
            file_basename = os.path.basename(file_path).lower()
            abs_path = os.path.abspath(file_path).lower()

            # PDFs: allow known report names OR anything saved under reports/ (but never sessions/).
            is_report_pdf = (
                file_ext == '.pdf'
                and (
                    'report' in file_basename
                    or '\\reports\\' in abs_path
                    or '/reports/' in abs_path
                )
                and '\\reports\\sessions\\' not in abs_path
                and '/reports/sessions/' not in abs_path
            )

            # JSON: allow known report payload names and report folders.
            json_name_keywords = (
                'report', 'metric', 'ecg_data', 'payload', 'unified', 'hrv', 'hyper'
            )
            is_report_json_name = file_ext == '.json' and any(k in file_basename for k in json_name_keywords)
            is_report_json_path = (
                file_ext == '.json'
                and (
                    '\\reports\\' in abs_path
                    or '/reports/' in abs_path
                )
                and 'upload_log.json' not in file_basename
                and '\\reports\\sessions\\' not in abs_path
                and '/reports/sessions/' not in abs_path
            )

            if not (is_report_pdf or is_report_json_name or is_report_json_path):
                return {
                    "status": "skipped",
                    "message": f"File {file_basename} is not a report or metric file - not uploaded"
                }
            
            # Prepare metadata - ONLY include essential report information
            upload_metadata = {
                "filename": os.path.basename(file_path),
                "uploaded_at": datetime.now().isoformat(),
                "file_size": os.path.getsize(file_path),
                "file_type": Path(file_path).suffix,
            }
            
            # Only add specific metadata fields if provided
            if metadata:
                # Include all patient details and metrics in metadata
                allowed_keys = [
                    'patient_name', 'patient_age', 'patient_gender', 'patient_address', 
                    'patient_phone', 'master_phone', 'login_identifier', 'report_date', 'machine_serial',
                    'heart_rate', 'pr_interval', 'qrs_duration', 'qt_interval', 
                    'qtc_interval', 'st_segment', 'report_type'
                ]
                filtered_metadata = {k: v for k, v in metadata.items() if k in allowed_keys}
                upload_metadata.update(filtered_metadata)
            
            # Check if online - if offline, queue for later upload
            if self.offline_queue and not self.offline_queue.is_online():
                # Queue for upload when internet is restored
                queue_payload = {
                    'file_path': file_path,
                    'metadata': upload_metadata,
                    'cloud_service': self.cloud_service
                }
                self.offline_queue.queue_data('cloud_report', queue_payload, priority=2)  # High priority
                print(f"[QUEUE] Queued report for upload when online: {filename}")
                return {
                    "status": "queued",
                    "message": f"Report queued for upload when internet connection is restored",
                    "filename": filename
                }
            
            # Upload based on configured service
            if self.cloud_service == 's3':
                result = self._upload_to_s3(file_path, upload_metadata)
            elif self.cloud_service == 'azure':
                result = self._upload_to_azure(file_path, upload_metadata)
            elif self.cloud_service == 'gcs':
                result = self._upload_to_gcs(file_path, upload_metadata)
            elif self.cloud_service == 'api':
                result = self._upload_to_api(file_path, upload_metadata)
            elif self.cloud_service == 'ftp':
                result = self._upload_to_ftp(file_path, upload_metadata, use_sftp=False)
            elif self.cloud_service == 'sftp':
                result = self._upload_to_ftp(file_path, upload_metadata, use_sftp=True)
            elif self.cloud_service == 'dropbox':
                result = self._upload_to_dropbox(file_path, upload_metadata)
            else:
                result = {"status": "error", "message": f"Unknown cloud service: {self.cloud_service}"}
            
            # If upload failed and we have offline queue, queue for retry
            if result.get("status") != "success" and self.offline_queue:
                queue_payload = {
                    'file_path': file_path,
                    'metadata': upload_metadata,
                    'cloud_service': self.cloud_service
                }
                self.offline_queue.queue_data('cloud_report', queue_payload, priority=2)
                result["status"] = "queued"
                result["message"] = "Upload failed - queued for retry when online"
            
            # Log the upload
            if result.get("status") == "success":
                self._log_upload(file_path, result, upload_metadata)
            
            return result
            
        except Exception as e:
            # On any error, try to queue if offline queue is available
            if self.offline_queue:
                try:
                    upload_metadata = {
                        "filename": os.path.basename(file_path),
                        "uploaded_at": datetime.now().isoformat(),
                        "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                        "file_type": Path(file_path).suffix,
                    }
                    if metadata:
                        upload_metadata.update(metadata)
                    
                    queue_payload = {
                        'file_path': file_path,
                        'metadata': upload_metadata,
                        'cloud_service': self.cloud_service
                    }
                    self.offline_queue.queue_data('cloud_report', queue_payload, priority=2)
                    return {
                        "status": "queued",
                        "message": f"Error occurred - queued for upload when online: {str(e)}"
                    }
                except:
                    pass
            
            return {"status": "error", "message": str(e)}
    
    def upload_user_signup(self, user_data):
        """
        Upload user signup details to cloud storage
        Prevents duplicate uploads - checks if user has already been uploaded
        Supports offline mode - queues for upload when internet is restored
        
        Args:
            user_data (dict): Dictionary containing user signup information
                             {username, full_name, age, gender, phone, address, serial_number, registered_at}
        
        Returns:
            dict: Upload result with status and details
        """
        print(f"[INFO] upload_user_signup called with data: {user_data}")
        
        if not self.upload_enabled:
            msg = "Cloud upload is disabled"
            print(f"[ERR] {msg}")
            return {"status": "disabled", "message": msg}
        
        if not self.is_configured():
            msg = f"Cloud service '{self.cloud_service}' is not properly configured"
            print(f"[ERR] {msg}")
            return {"status": "error", "message": msg}
        
        if not user_data or not isinstance(user_data, dict):
            msg = "Invalid user data"
            print(f"[ERR] {msg}")
            return {"status": "error", "message": msg}
        
        # Check if this user has already been uploaded
        username = user_data.get('username', 'unknown')
        serial_number = user_data.get('serial_number', '')
        
        try:
            if os.path.exists(self.upload_log_path):
                with open(self.upload_log_path, 'r') as f:
                    log_data = json.load(f)
                
                # Check if user with same username or serial has been uploaded
                for entry in log_data:
                    metadata = entry.get('metadata', {})
                    if metadata.get('type') == 'user_signup':
                        if (metadata.get('username') == username or 
                            (serial_number and metadata.get('serial_number') == serial_number)):
                            if entry.get('result', {}).get('status') == 'success':
                                print(f"[INFO] User signup for '{username}' already uploaded - skipping duplicate")
                                return {
                                    "status": "already_uploaded",
                                    "message": f"User signup for '{username}' has already been uploaded",
                                    "username": username
                                }
        except Exception as e:
            print(f"[WARN] Error checking user signup history: {e}")
        
        # Check if online - if offline, queue for later upload
        if self.offline_queue and not self.offline_queue.is_online():
            queue_payload = {
                'user_data': user_data,
                'cloud_service': self.cloud_service
            }
            self.offline_queue.queue_data('cloud_user_signup', queue_payload, priority=1)  # Highest priority
            print(f"[QUEUE] Queued user signup for upload when online: {username}")
            return {
                "status": "queued",
                "message": f"User signup queued for upload when internet connection is restored",
                "username": username
            }
        
        try:
            # Create a JSON file with user signup details
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"user_signup_{username}_{timestamp}.json"
            
            # Create temp directory if it doesn't exist
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, filename)
            
            print(f"📝 Creating user signup file: {file_path}")
            
            # Add timestamp to user data
            upload_data = user_data.copy()
            upload_data['uploaded_at'] = datetime.now().isoformat()
            
            # Write user data to JSON file
            with open(file_path, 'w') as f:
                json.dump(upload_data, f, indent=2)
            
            print(f"[OK] User signup file created successfully")
            
            # Upload to cloud
            metadata = {
                'type': 'user_signup',
                'username': username,
                'serial_number': serial_number,
                'master_phone': str(user_data.get('phone', '') or user_data.get('contact', '') or ''),
                'uploaded_at': datetime.now().isoformat()
            }
            
            print(f"☁️ Uploading to {self.cloud_service}...")
            
            result = None
            if self.cloud_service == 's3':
                result = self._upload_to_s3(file_path, metadata)
            elif self.cloud_service == 'azure':
                result = self._upload_to_azure(file_path, metadata)
            elif self.cloud_service == 'gcs':
                result = self._upload_to_gcs(file_path, metadata)
            elif self.cloud_service == 'api':
                result = self._upload_to_api(file_path, metadata)
            elif self.cloud_service == 'ftp':
                result = self._upload_to_ftp(file_path, metadata)
            elif self.cloud_service == 'sftp':
                result = self._upload_to_ftp(file_path, metadata, use_sftp=True)
            elif self.cloud_service == 'dropbox':
                result = self._upload_to_dropbox(file_path, metadata)
            else:
                result = {"status": "error", "message": f"Unknown cloud service: {self.cloud_service}"}
            
            print(f"📤 Upload result: {result}")
            
            # Clean up temp file
            try:
                os.remove(file_path)
                print(f"🗑️ Temp file removed: {file_path}")
            except Exception as cleanup_err:
                print(f"[WARN] Could not remove temp file: {cleanup_err}")
            
            # Log upload
            if result and result.get("status") == "success":
                self._log_upload(filename, result, metadata)
                print(f"[OK] User signup uploaded to {self.cloud_service}: {username}")
            else:
                print(f"[ERR] Upload failed: {result}")
            
            return result
            
        except Exception as e:
            import traceback
            error_msg = f"Failed to upload user signup: {str(e)}"
            print(f"[ERR] {error_msg}")
            print(f"Stack trace: {traceback.format_exc()}")
            return {"status": "error", "message": error_msg}

    def get_available_doctors(self, force_refresh=False):
        """
        Fetch list of available doctors from the API.
        Returns a list of doctor names.
        Uses a cache to avoid redundant API calls.
        """
        import time
        default_doctors = []
        
        # Return cache if available and not forced refresh (cache for 1 hour)
        now = time.time()
        if not force_refresh and self._doctor_list_cache and (now - self._last_doctor_fetch_time < 3600):
            return self._doctor_list_cache

        if not self.doctor_review_api_url:
            return self._doctor_list_cache or default_doctors
        
        try:
            url = self.doctor_review_api_url.rstrip('/')
            
            print(f"🔹 Fetching doctor list from {url}")
            
            headers = {}
            if self.doctor_review_api_key:
                # Use x-api-key for AWS API Gateway. 
                # Avoid 'Authorization' header as it triggers SigV4 validation errors on AWS.
                headers['x-api-key'] = self.doctor_review_api_key

            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                doctors = []
                
                if isinstance(data, dict):
                    if isinstance(data.get("names"), list):
                        doctors.extend([str(name).strip() for name in data["names"] if str(name).strip()])
                    doctor_data = data.get("doctors", data)
                else:
                    doctor_data = data

                if isinstance(doctor_data, list):
                    for item in doctor_data:
                        if isinstance(item, str):
                            doctors.append(item.strip())
                        elif isinstance(item, dict):
                            name = (
                                item.get('doctor_name')
                                or item.get('name')
                                or item.get('doctorName')
                                or item.get('username')
                            )
                            if name:
                                doctors.append(str(name).strip())

                doctors = [name for name in doctors if name]
                doctors = list(dict.fromkeys(doctors))
                
                if doctors:
                    print(f"[OK] Fetched {len(doctors)} doctors from API")
                    self._doctor_list_cache = doctors
                    self._last_doctor_fetch_time = now
                    return doctors
            
            print(f"[WARN] Failed to fetch doctors (Status {response.status_code}). Using fallback.")
            
        except Exception as e:
            print(f"[WARN] Error fetching doctor list: {e}. Using fallback.")
            
        # If fetch failed but we have a stale cache, return it
        if self._doctor_list_cache:
            return self._doctor_list_cache
            
        return self._doctor_list_cache or default_doctors
    
    def _upload_to_s3(self, file_path, metadata):
        """Upload to AWS S3"""
        try:
            import boto3
            from botocore.exceptions import ClientError
            from botocore.config import Config
            from boto3.s3.transfer import TransferConfig

            retry_max = max(3, self._safe_int_env("S3_UPLOAD_MAX_RETRIES", 8))
            connect_timeout = max(1.0, self._safe_float_env("S3_CONNECT_TIMEOUT", 5.0))
            read_timeout = max(5.0, self._safe_float_env("S3_READ_TIMEOUT", 60.0))
            multipart_threshold_mb = max(5, self._safe_int_env("S3_MULTIPART_THRESHOLD_MB", 8))
            multipart_chunksize_mb = max(5, self._safe_int_env("S3_MULTIPART_CHUNKSIZE_MB", 8))
            max_concurrency = max(2, self._safe_int_env("S3_MAX_CONCURRENCY", 6))
            
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.s3_region,
                config=Config(
                    retries={"max_attempts": retry_max, "mode": "adaptive"},
                    connect_timeout=connect_timeout,
                    read_timeout=read_timeout,
                    s3={"addressing_style": "virtual"},
                ),
            )
            
            # Generate S3 key
            filename = os.path.basename(file_path)
            s3_key = self._build_s3_key(filename, metadata)
            transfer_cfg = TransferConfig(
                multipart_threshold=multipart_threshold_mb * 1024 * 1024,
                multipart_chunksize=multipart_chunksize_mb * 1024 * 1024,
                max_concurrency=max_concurrency,
                use_threads=True,
            )
            
            # Compress JSON files using gzip to reduce size (MB -> KB)
            temp_gzipped_path = None
            if file_path.lower().endswith('.json'):
                try:
                    import gzip
                    import tempfile
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_content = f.read()
                    json_bytes = json_content.encode('utf-8')
                    with tempfile.NamedTemporaryFile(suffix='.json.gz', delete=False) as tf:
                        with gzip.GzipFile(fileobj=tf, mode='wb') as gz:
                            gz.write(json_bytes)
                        temp_gzipped_path = tf.name
                except Exception as e:
                    print(f"[WARN] Failed to compress JSON file {file_path}: {e}")
                    temp_gzipped_path = None

            extra_args = {'Metadata': {k: str(v) for k, v in metadata.items()}}
            if temp_gzipped_path:
                upload_src = temp_gzipped_path
                extra_args['ContentType'] = 'application/json'
                extra_args['ContentEncoding'] = 'gzip'
            else:
                upload_src = file_path

            try:
                # Upload file
                s3_client.upload_file(
                    upload_src,
                    self.s3_bucket,
                    s3_key,
                    ExtraArgs=extra_args,
                    Config=transfer_cfg,
                )
            finally:
                if temp_gzipped_path:
                    try:
                        os.remove(temp_gzipped_path)
                    except Exception:
                        pass
            
            # Generate presigned URL (optional)
            url = f"https://{self.s3_bucket}.s3.{self.s3_region}.amazonaws.com/{s3_key}"
            
            return {
                "status": "success",
                "service": "s3",
                "url": url,
                "key": s3_key,
                "bucket": self.s3_bucket
            }
            
        except ImportError:
            return {"status": "error", "message": "boto3 not installed. Run: pip install boto3"}
        except ClientError as e:
            return {"status": "error", "message": f"S3 upload failed: {str(e)}"}
    
    def upload_complete_report_package(self, pdf_path, patient_data, ecg_data_file, report_metadata=None, report_type="12_LEAD_ECG", clinical_measurements=None):
        """
        Upload complete ECG report package to S3:
        - PDF report (kept separate)
        - Unified report_package.json containing:
          - device_provenance
          - doctor_registration
          - patient
          - clinical_measurements
          - ecg_data
          - report_metadata
        Supports offline mode - queues for upload when internet is restored
        """
        if not self.upload_enabled:
            return {"status": "disabled", "message": "Cloud upload is disabled"}
            
        if not self.is_configured():
            return {"status": "error", "message": f"Cloud service '{self.cloud_service}' is not properly configured"}
        
        if self.cloud_service != 's3':
            return {"status": "error", "message": "Complete report package upload only supports S3"}
        
        # Check if online - if offline, queue for later upload
        if self.offline_queue and not self.offline_queue.is_online():
            print(f"[OFFLINE] Queuing complete report package for upload when online")
            queue_payload = {
                "pdf_path": pdf_path,
                "patient_data": patient_data,
                "ecg_data_file": ecg_data_file,
                "report_metadata": report_metadata,
                "report_type": report_type,
                "clinical_measurements": clinical_measurements
            }
            self.offline_queue.queue_data("cloud_complete_package", queue_payload, priority=2)
            return {
                "status": "queued",
                "message": "Report package queued for upload when internet connection is restored"
            }
        
        try:
            import boto3
            from botocore.exceptions import ClientError
            import uuid
            import platform
            from utils.license_manager import load_registration_profile, load_token_file, get_machine_context
            
            # Helper to sanitize numbers
            def _to_num(val):
                if val is None:
                    return None
                try:
                    if isinstance(val, (int, float)):
                        return val
                    s = str(val).strip().replace("°", "").replace(" ms", "").replace(" bpm", "")
                    if not s or s == "--":
                        return None
                    if "." in s:
                        return float(s)
                    return int(s)
                except Exception:
                    return None

            profile = load_registration_profile()
            token = load_token_file()
            machine_ctx = get_machine_context()
            
            # Parse ECG JSON
            ecg_json = {}
            if ecg_data_file:
                if isinstance(ecg_data_file, dict):
                    ecg_json = ecg_data_file
                elif isinstance(ecg_data_file, str) and os.path.exists(ecg_data_file):
                    try:
                        with open(ecg_data_file, 'r', encoding='utf-8') as f:
                            ecg_json = json.load(f)
                    except Exception as e:
                        print(f"[WARN] Could not parse ECG data file {ecg_data_file}: {e}")
            
            # Build clinical measurements
            if not clinical_measurements:
                meta = report_metadata or {}
                clinical_measurements = {
                    "heart_rate": _to_num(meta.get("heart_rate") or meta.get("heart_rate_bpm") or meta.get("heart_rate_avg") or meta.get("HR_bpm") or meta.get("Heart_Rate")),
                    "pr_interval": _to_num(meta.get("pr_interval") or meta.get("pr_ms")),
                    "qrs_duration": _to_num(meta.get("qrs_duration") or meta.get("qrs_ms")),
                    "qt_interval": _to_num(meta.get("qt_interval") or meta.get("qt_ms")),
                    "qtc": _to_num(meta.get("qtc_interval") or meta.get("qtc_ms") or meta.get("qtc")),
                    "p_axis": _to_num(meta.get("p_axis") or meta.get("p_axis_display") or meta.get("p_axis_deg")),
                    "qrs_axis": _to_num(meta.get("qrs_axis") or meta.get("qrs_axis_display") or meta.get("qrs_axis_deg")),
                    "t_axis": _to_num(meta.get("t_axis") or meta.get("t_axis_display") or meta.get("t_axis_deg")),
                }
            # Clean clinical measurements to keep only valid values
            clinical_measurements = {k: v for k, v in clinical_measurements.items() if v is not None}
            
            # Generate/retrieve report ID and timestamps
            report_id = (report_metadata or {}).get("report_id") or str(uuid.uuid4())
            generated_at = (report_metadata or {}).get("generated_at") or (report_metadata or {}).get("report_date") or datetime.utcnow().isoformat()
            uploaded_at = datetime.utcnow().isoformat()
            
            # Build device provenance
            device_provenance = {
                "rhythmultra_serial": profile.get("rhythmultra_serial") or profile.get("rhythmulta_serial") or profile.get("RhythmUltra_serial") or "",
                "machine_serial_id": machine_ctx.get("machine_serial_id") or profile.get("machine_serial_id") or "",
                "pc_name": machine_ctx.get("machine_name") or platform.node(),
                "software_version": "2.0.0"
            }
            
            # Extract license_id and seat_number
            license_id = (token or {}).get("license_id") or (token or {}).get("id") or profile.get("license_id") or ""
            seat_number = (token or {}).get("seat_number") or (token or {}).get("seat") or profile.get("seat_number") or 1
            
            # Add seat number to device provenance/rhythmultra if needed
            rhythmultra = {
                "serial_number": device_provenance["rhythmultra_serial"],
                "seat_number": seat_number
            }
            
            # Build unified report package JSON
            upload_package = {
                "report_type": report_type,
                "device_provenance": device_provenance,
                "rhythmultra": rhythmultra,
                "doctor_registration": {
                    "doctor_name": profile.get("doctor_name", ""),
                    "hospital_name": profile.get("hospital_name", ""),
                    "hospital_address": profile.get("hospital_address", ""),
                    "phone": profile.get("phone", "")
                },
                "patient": patient_data or {},
                "clinical_measurements": clinical_measurements,
                "ecg_data": ecg_json,
                "report_metadata": {
                    "report_id": report_id,
                    "generated_at": generated_at,
                    "uploaded_at": uploaded_at,
                    "report_version": "1",
                    "license_id": license_id
                }
            }
            
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.s3_region
            )
            
            import re
            patient_name = "Unknown"
            if patient_data:
                patient_name = (
                    patient_data.get("name") 
                    or patient_data.get("patient_name") 
                    or f"{patient_data.get('first_name', '')} {patient_data.get('last_name', '')}".strip()
                )
            if not patient_name or patient_name == "Unknown":
                if report_metadata:
                    patient_name = report_metadata.get("patient_name") or report_metadata.get("name") or "Unknown"
            
            sanitized_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(patient_name))
            sanitized_name = re.sub(r'_+', '_', sanitized_name).strip('_')
            if not sanitized_name:
                sanitized_name = "Unknown"

            # S3 prefix and keys with patient name and report_id for easy identification.
            # The device folder and date come from the report's own filename identity
            # so the review Lambda can find it — see _report_location().
            pdf_filename = os.path.basename(pdf_path) if pdf_path else f"{sanitized_name}_{report_id}.pdf"
            machine_serial_dir, date_path = self._report_location(pdf_filename)
            s3_prefix = f"reports/{date_path}/{machine_serial_dir}/{report_id}"
            package_s3_key = f"{s3_prefix}/{sanitized_name}_{report_id}.json"
            pdf_s3_key = f"{s3_prefix}/{pdf_filename}"

            
            # Convenience custom metadata headers (Step 7)
            custom_metadata = {
                "report_type": report_type,
                "rhythmultra_serial": device_provenance["rhythmultra_serial"],
                "doctor_name": profile.get("doctor_name", ""),
                "hospital_name": profile.get("hospital_name", "")
            }
            # Clean headers values to be strings
            s3_metadata_headers = {k: str(v) for k, v in custom_metadata.items() if v is not None}
            
            upload_results = {
                "status": "success",
                "service": "s3",
                "bucket": self.s3_bucket,
                "uploads": []
            }
            
            # 1. Upload PDF Report
            if pdf_path and os.path.exists(pdf_path):
                s3_client.upload_file(
                    pdf_path,
                    self.s3_bucket,
                    pdf_s3_key,
                    ExtraArgs={'Metadata': s3_metadata_headers}
                )
                upload_results["uploads"].append({
                    "type": "pdf_report",
                    "key": pdf_s3_key,
                    "url": f"https://{self.s3_bucket}.s3.{self.s3_region}.amazonaws.com/{pdf_s3_key}"
                })
                print(f"[OK] Uploaded PDF report to S3: {pdf_s3_key}")
            
            # 2. Upload Package JSON
            def _sanitize(o):
                try:
                    import numpy as np
                except ImportError:
                    np = None
                
                if isinstance(o, dict):
                    return {k: _sanitize(v) for k, v in o.items()}
                elif isinstance(o, (list, tuple)):
                    return [_sanitize(x) for x in o]
                elif np is not None:
                    if isinstance(o, np.integer):
                        return int(o)
                    elif isinstance(o, np.floating):
                        return float(o)
                    elif isinstance(o, np.bool_):
                        return bool(o)
                    elif isinstance(o, np.ndarray):
                        return [_sanitize(x) for x in o.tolist()]
                return o

            upload_package = _sanitize(upload_package)

            import tempfile
            import gzip

            # Dump as compact JSON string (no indent, no extra spacing to save space)
            package_json_str = json.dumps(upload_package, ensure_ascii=False, separators=(',', ':'))
            package_json_bytes = package_json_str.encode('utf-8')

            # Write as gzipped bytes to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.json.gz', delete=False) as f:
                with gzip.GzipFile(fileobj=f, mode='wb') as gz:
                    gz.write(package_json_bytes)
                package_file_path = f.name
            
            try:
                # Upload with gzip Content-Encoding so S3 clients/browsers automatically
                # decompress it transparently upon download. Stored size is reduced by ~93% (MB -> KB).
                s3_client.upload_file(
                    package_file_path,
                    self.s3_bucket,
                    package_s3_key,
                    ExtraArgs={
                        'Metadata': s3_metadata_headers,
                        'ContentType': 'application/json',
                        'ContentEncoding': 'gzip'
                    }
                )
                upload_results["uploads"].append({
                    "type": "package_json",
                    "key": package_s3_key,
                    "url": f"https://{self.s3_bucket}.s3.{self.s3_region}.amazonaws.com/{package_s3_key}"
                })
                print(f"[OK] Uploaded report package JSON to S3: {package_s3_key}")
            finally:
                try:
                    os.remove(package_file_path)
                except:
                    pass
            
            # Log the upload
            if upload_results["uploads"]:
                self._log_upload(pdf_path or "package.json", upload_results, {
                    "type": "complete_report_package",
                    "timestamp": now.strftime("%Y%m%d_%H%M%S"),
                    "report_id": report_id,
                    "report_type": report_type
                })
                
            return upload_results
            
        except ImportError as e:
            return {"status": "error", "message": f"boto3 not installed: {str(e)}"}
        except ClientError as e:
            return {"status": "error", "message": f"S3 upload failed: {str(e)}"}
        except Exception as e:
            import traceback
            print(f"[ERR] S3 unified package upload error: {e}")
            traceback.print_exc()
            return {"status": "error", "message": f"Upload failed: {str(e)}"}

    def list_reports(self, prefix: str = "reports/"):
        """List report objects in S3 (PDF and JSON under prefix)."""
        if not (self.upload_enabled and self.cloud_service == 's3' and self.s3_bucket):
            return {"status": "error", "message": "S3 not configured"}
        try:
            import boto3
            s3 = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.s3_region
            )
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix)
            items = []
            for page in pages:
                for obj in page.get('Contents', []) or []:
                    key = obj['Key']
                    if not key.lower().endswith(('.pdf', '.json')):
                        continue
                    items.append({
                        'key': key,
                        'size': obj.get('Size', 0),
                        'last_modified': obj.get('LastModified').isoformat() if obj.get('LastModified') else '',
                        'url': f"https://{self.s3_bucket}.s3.{self.s3_region}.amazonaws.com/{key}"
                    })
            return {"status": "success", "items": items}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def generate_presigned_url(self, key: str, expires_in: int = 3600):
        """Generate a presigned URL for a given S3 object key."""
        try:
            import boto3
            s3 = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.s3_region
            )
            url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.s3_bucket, 'Key': key},
                ExpiresIn=expires_in
            )
            return {"status": "success", "url": url}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def delete_file(self, key: str):
        """Delete a file from S3 bucket"""
        try:
            import boto3
            s3 = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                region_name=self.s3_region
            )
            
            # Delete the object
            s3.delete_object(Bucket=self.s3_bucket, Key=key)
            print(f"[OK] Deleted from S3: {key}")
            return {"status": "success", "message": f"Deleted {key}"}
            
        except Exception as e:
            print(f"[ERR] S3 deletion error for {key}: {e}")
            return {"status": "error", "message": str(e)}
    
    def _upload_to_azure(self, file_path, metadata):
        """Upload to Azure Blob Storage"""
        try:
            from azure.storage.blob import BlobServiceClient
            
            blob_service_client = BlobServiceClient.from_connection_string(self.azure_connection_string)
            container_client = blob_service_client.get_container_client(self.azure_container)
            
            # Ensure container exists
            try:
                container_client.create_container()
            except Exception:
                pass  # Container already exists
            
            # Generate blob name
            filename = os.path.basename(file_path)
            timestamp = datetime.now().strftime("%Y/%m/%d")
            blob_name = f"ecg-reports/{timestamp}/{filename}"
            
            # Upload file
            blob_client = container_client.get_blob_client(blob_name)
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True, metadata=metadata)
            
            url = blob_client.url
            
            return {
                "status": "success",
                "service": "azure",
                "url": url,
                "blob_name": blob_name,
                "container": self.azure_container
            }
            
        except ImportError:
            return {"status": "error", "message": "azure-storage-blob not installed. Run: pip install azure-storage-blob"}
        except Exception as e:
            return {"status": "error", "message": f"Azure upload failed: {str(e)}"}
    
    def _upload_to_gcs(self, file_path, metadata):
        """Upload to Google Cloud Storage"""
        try:
            from google.cloud import storage
            
            storage_client = storage.Client.from_service_account_json(self.gcs_credentials_path)
            bucket = storage_client.bucket(self.gcs_bucket)
            
            # Generate blob name
            filename = os.path.basename(file_path)
            timestamp = datetime.now().strftime("%Y/%m/%d")
            blob_name = f"ecg-reports/{timestamp}/{filename}"
            
            # Upload file
            blob = bucket.blob(blob_name)
            blob.metadata = metadata
            blob.upload_from_filename(file_path)
            
            url = blob.public_url
            
            return {
                "status": "success",
                "service": "gcs",
                "url": url,
                "blob_name": blob_name,
                "bucket": self.gcs_bucket
            }
            
        except ImportError:
            return {"status": "error", "message": "google-cloud-storage not installed. Run: pip install google-cloud-storage"}
        except Exception as e:
            return {"status": "error", "message": f"GCS upload failed: {str(e)}"}
    
    def _upload_to_api(self, file_path, metadata):
        """Upload to custom API endpoint"""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                headers = {}
                if self.api_key:
                    headers['Authorization'] = f'Bearer {self.api_key}'
                
                data = {'metadata': json.dumps(metadata)}
                
                response = requests.post(
                    self.api_endpoint,
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json() if response.content else {}
                    return {
                        "status": "success",
                        "service": "api",
                        "response": result,
                        "url": result.get('url', self.api_endpoint)
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"API returned status {response.status_code}: {response.text}"
                    }
                    
        except Exception as e:
            return {"status": "error", "message": f"API upload failed: {str(e)}"}
    
    def _upload_to_ftp(self, file_path, metadata, use_sftp=False):
        """Upload to FTP/SFTP server"""
        try:
            if use_sftp:
                import paramiko
                
                transport = paramiko.Transport((self.ftp_host, self.ftp_port))
                transport.connect(username=self.ftp_username, password=self.ftp_password)
                sftp = paramiko.SFTPClient.from_transport(transport)
                
                # Create remote directory if needed
                remote_file = f"{self.ftp_remote_path}/{os.path.basename(file_path)}"
                sftp.put(file_path, remote_file)
                sftp.close()
                transport.close()
                
            else:
                from ftplib import FTP
                
                ftp = FTP()
                ftp.connect(self.ftp_host, self.ftp_port)
                ftp.login(self.ftp_username, self.ftp_password)
                
                # Upload file
                with open(file_path, 'rb') as f:
                    remote_file = f"{self.ftp_remote_path}/{os.path.basename(file_path)}"
                    ftp.storbinary(f'STOR {remote_file}', f)
                
                ftp.quit()
            
            return {
                "status": "success",
                "service": "sftp" if use_sftp else "ftp",
                "remote_path": remote_file
            }
            
        except ImportError:
            return {"status": "error", "message": "paramiko not installed for SFTP. Run: pip install paramiko"}
        except Exception as e:
            return {"status": "error", "message": f"FTP upload failed: {str(e)}"}
    
    def _upload_to_dropbox(self, file_path, metadata):
        """Upload to Dropbox"""
        try:
            import dropbox
            
            dbx = dropbox.Dropbox(self.dropbox_token)
            
            # Generate Dropbox path
            filename = os.path.basename(file_path)
            timestamp = datetime.now().strftime("%Y/%m/%d")
            dropbox_path = f"/ECG-Reports/{timestamp}/{filename}"
            
            # Upload file
            with open(file_path, 'rb') as f:
                dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite)
            
            # Get shareable link
            try:
                link = dbx.sharing_create_shared_link(dropbox_path)
                url = link.url
            except:
                url = dropbox_path
            
            return {
                "status": "success",
                "service": "dropbox",
                "url": url,
                "path": dropbox_path
            }
            
        except ImportError:
            return {"status": "error", "message": "dropbox not installed. Run: pip install dropbox"}
        except Exception as e:
            return {"status": "error", "message": f"Dropbox upload failed: {str(e)}"}
    
    def _log_upload(self, file_path, result, metadata):
        """Log successful upload to tracking file"""
        try:
            # Load existing log
            log_data = []
            if os.path.exists(self.upload_log_path):
                try:
                    with open(self.upload_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        raw = f.read().strip()
                    log_data = json.loads(raw) if raw else []
                except Exception:
                    log_data = []
            
            # Add new entry
            log_entry = {
                "local_path": file_path,
                "uploaded_at": datetime.now().isoformat(),
                "service": self.cloud_service,
                "result": result,
                "metadata": metadata
            }
            log_data.append(log_entry)
            
            # Save log
            os.makedirs(os.path.dirname(self.upload_log_path), exist_ok=True)
            with open(self.upload_log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not log upload: {e}")
    
    def get_upload_history(self, limit=50):
        """Get recent upload history"""
        try:
            if os.path.exists(self.upload_log_path):
                with open(self.upload_log_path, 'r') as f:
                    log_data = json.load(f)
                return log_data[-limit:]
            return []
        except Exception:
            return []
    
    def get_uploaded_files_list(self):
        """
        Get a list of all uploaded filenames for easy checking
        
        Returns:
            list: List of successfully uploaded filenames
        """
        try:
            if os.path.exists(self.upload_log_path):
                with open(self.upload_log_path, 'r') as f:
                    log_data = json.load(f)
                
                uploaded_files = []
                for entry in log_data:
                    if entry.get('result', {}).get('status') == 'success':
                        filename = entry.get('metadata', {}).get('filename', '')
                        if not filename:
                            filename = os.path.basename(entry.get('local_path', ''))
                        if filename:
                            uploaded_files.append(filename)
                
        except Exception as e:
            print(f"Warning: Could not log upload: {e}")
    
    def get_upload_history(self, limit=50):
        """Get recent upload history"""
        try:
            if os.path.exists(self.upload_log_path):
                with open(self.upload_log_path, 'r') as f:
                    log_data = json.load(f)
                return log_data[-limit:]
            return []
        except Exception:
            return []
    
    def get_uploaded_files_list(self):
        """
        Get a list of all uploaded filenames for easy checking
        
        Returns:
            list: List of successfully uploaded filenames
        """
        try:
            if os.path.exists(self.upload_log_path):
                with open(self.upload_log_path, 'r') as f:
                    log_data = json.load(f)
                
                uploaded_files = []
                for entry in log_data:
                    if entry.get('result', {}).get('status') == 'success':
                        filename = entry.get('metadata', {}).get('filename', '')
                        if not filename:
                            filename = os.path.basename(entry.get('local_path', ''))
                        if filename:
                            uploaded_files.append(filename)
                
                return uploaded_files
            return []
        except Exception as e:
            print(f"[WARN] Error getting uploaded files list: {e}")
            return []
    
    def clear_upload_log(self):
        """
        Clear the upload log (use with caution!)
        This will allow re-uploading of all files
        
        Returns:
            dict: Result with status
        """
        try:
            if os.path.exists(self.upload_log_path):
                # Backup the log before clearing
                backup_path = f"{self.upload_log_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                
                # Copy to backup
                import shutil
                shutil.copy2(self.upload_log_path, backup_path)
                
                # Clear the log
                with open(self.upload_log_path, 'w') as f:
                    json.dump([], f)
                
                return {
                    "status": "success",
                    "message": f"Upload log cleared. Backup saved to {backup_path}"
                }
            else:
                return {
                    "status": "success",
                    "message": "Upload log does not exist - nothing to clear"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to clear upload log: {str(e)}"
            }
    
    def get_registered_doctors(self):
        """Fetch active doctor list from the AWS doctor endpoint."""
        try:
            url = self.doctor_review_api_url or "https://6jhix49qt6.execute-api.us-east-1.amazonaws.com/api/doctor/upload"
            headers = {}
            if self.doctor_review_api_key:
                headers['x-api-key'] = self.doctor_review_api_key
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("doctors") or data.get("data") or []
        except Exception as e:
            print(f"[CloudUploader] Error", "No Internet Connection is available, Could not fetch available doctors list.: {e}")
        return []

    def send_for_doctor_review(self, file_path, doctor_name, metadata=None):
        """
        Send ECG/Analysis report for doctor review.

        Supports BOTH Lambda versions transparently:
          • New Lambda (JSON mode):  POST JSON → get presigned uploadUrl → PUT file to S3
          • Old Lambda (multipart):  POST multipart/form-data with PDF in body

        The method tries JSON first. If the server replies with
        "Content-Type must be multipart/form-data" it automatically retries as multipart.

        Args:
            file_path (str): Path to the PDF report
            doctor_name (str): Name of the doctor to review
            metadata (dict): Optional metadata (report_type, patient_name, etc.)

        Returns:
            dict: Result with status and message
        """
        base_url = self.doctor_review_api_url
        if not base_url:
            base_url = "https://6jhix49qt6.execute-api.us-east-1.amazonaws.com/api/doctor/upload"

        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File not found: {file_path}"}

        # Check if online
        if self.offline_queue and not self.offline_queue.is_online():
            self.offline_queue.queue_data('doctor_review_v2', {
                'file_path': file_path, 'doctor_name': doctor_name, 'metadata': metadata
            }, priority=1)
            return {"status": "queued", "message": "Queued for doctor review (offline)"}

        try:
            endpoint = base_url.rstrip('/')
            meta = metadata or {}
            patient_name = meta.get('patient_name') or "Unknown"
            report_type  = (meta.get('report_type') or "ECG").lower()
            filename     = os.path.basename(file_path)

            # ── Auto-resolve doctor name against registered doctor list ────────
            resolved_doctor = doctor_name
            try:
                doctors = self.get_registered_doctors() or []
                clean_target = re.sub(r"^dr[._\s]*", "", str(doctor_name or "").strip().lower()).replace("_", "").replace(" ", "")
                for doc in doctors:
                    dname = str(doc.get("doctor_name", "") or "")
                    clean_doc = re.sub(r"^dr[._\s]*", "", dname.strip().lower()).replace("_", "").replace(" ", "")
                    if clean_target and (clean_target == clean_doc or clean_target in clean_doc or clean_doc in clean_target):
                        resolved_doctor = dname
                        print(f"  → Resolved doctor name '{doctor_name}' → official API doctor '{resolved_doctor}'")
                        break
            except Exception as de:
                print(f"  → Doctor name resolution skipped: {de}")
            doctor_name = resolved_doctor

            print(f"🔹 Sending report for review → {endpoint} | doctor={doctor_name} | type={report_type} | file={filename}")



            # ── Resolve device serial ──────────────────────────────────────────
            RhythmUltra_serial = ""
            try:
                from utils.license_manager import load_token_file, get_RhythmUltra_serial
                token = load_token_file() or {}
                RhythmUltra_serial = (
                    token.get("rhythmultra_serial")
                    or token.get("RhythmUltra_serial")
                    or token.get("rhythmulta_serial")
                    or ""
                )
                if not RhythmUltra_serial:
                    RhythmUltra_serial = get_RhythmUltra_serial() or ""
            except Exception:
                RhythmUltra_serial = ""
            if not RhythmUltra_serial:
                RhythmUltra_serial = "DM ECG V1.0 A010"

            # The deviceId must be the one the report is filed under in S3, because
            # the Lambda matches the assignment against the stored report's device id
            # and its <DEVICE>_<DATE>_<TIME>.pdf key suffix. _report_location() is the
            # single source of that answer, so the upload and the assignment can never
            # disagree about which device a report belongs to.
            device_id_short, _report_date_path = self._report_location(filename)
            print(f"  → deviceId={device_id_short} (report filed under reports/{_report_date_path}/{device_id_short}/)")

            api_headers = {}
            if self.doctor_review_api_key:
                api_headers['x-api-key'] = self.doctor_review_api_key

            # Proactively ensure the PDF report package is uploaded to S3
            # so AWS S3 ingestion indexes the report into ecg_reports before doctor assignment.
            try:
                print("  → Ensuring complete report package is on S3 for doctor review DB indexing...")
                self.upload_complete_report_package(
                    pdf_path=file_path,
                    patient_data={"patient_name": patient_name, "name": patient_name},
                    ecg_data_file=None,
                    report_metadata=metadata or {},
                    report_type=report_type,
                )
            except Exception as _pro_e:
                print(f"  → Proactive upload attempt note: {_pro_e}")



            # ══════════════════════════════════════════════════════════════════
            # MODE A — JSON two-step flow (new Lambda)
            # ══════════════════════════════════════════════════════════════════
            json_body = {
                "fileName":   filename,
                "doctorName": doctor_name,
                "deviceId":   device_id_short,
                "reportType": report_type,
                "patientName": patient_name,
                "RhythmUltra_serial": RhythmUltra_serial,
            }
            print(f"  → [MODE-A] JSON POST: {json_body}")
            step1_resp = requests.post(
                endpoint,
                json=json_body,
                headers={**api_headers, "Content-Type": "application/json"},
                timeout=30,
            )
            print(f"  ← [MODE-A] status={step1_resp.status_code}  body={step1_resp.text[:300]}")

            # Check if the server wants multipart instead (old Lambda still deployed)
            _needs_multipart = False
            if step1_resp.status_code in (400, 415, 422):
                try:
                    _err_body = step1_resp.json()
                    _err_msg  = str(_err_body.get("message", "") or _err_body.get("error", "")).lower()
                    if "multipart" in _err_msg or "form-data" in _err_msg or "content-type" in _err_msg:
                        _needs_multipart = True
                        print("  → Server requires multipart — switching to MODE-B")
                except Exception:
                    if "multipart" in step1_resp.text.lower() or "form-data" in step1_resp.text.lower():
                        _needs_multipart = True
                        print("  → Server requires multipart (text match) — switching to MODE-B")

            if not _needs_multipart and step1_resp.status_code in (200, 201):
                # Parse presigned URL from JSON response
                try:
                    resp_data    = step1_resp.json()
                    presigned_url = resp_data.get("uploadUrl")
                except Exception:
                    presigned_url = None

                if presigned_url:
                    # ── Step 2: PUT file to presigned S3 URL ────────────────
                    print("  → [MODE-A] PUTting PDF to presigned S3 URL…")
                    with open(file_path, 'rb') as f:
                        put_resp = requests.put(
                            presigned_url,
                            data=f,
                            headers={"Content-Type": "application/pdf"},
                            timeout=120,
                        )
                    print(f"  ← [MODE-A] PUT status={put_resp.status_code}")
                    if put_resp.status_code in (200, 204):
                        print(f"[OK] Report sent via JSON/S3 presigned flow (assignment={resp_data.get('assignmentId')})")
                        self._log_upload(file_path, {"status": "success"},
                                         {"type": "doctor_review", "doctor": doctor_name,
                                          "assignmentId": resp_data.get("assignmentId", "")})
                        return {"status": "success", "message": "Report sent for review successfully"}
                    else:
                        return {"status": "error", "message": f"S3 upload failed (HTTP {put_resp.status_code})"}

                else:
                    # Some Lambda variants return direct success (no presigned URL)
                    try:
                        resp_data = step1_resp.json()
                    except Exception:
                        resp_data = {}
                    if resp_data.get("success") is True or resp_data.get("status") == "success":
                        self._log_upload(file_path, {"status": "success"},
                                         {"type": "doctor_review", "doctor": doctor_name})
                        return {"status": "success", "message": "Report sent for review successfully"}
                    # Fall through to MODE-B
                    _needs_multipart = True
                    print("  → No uploadUrl in JSON response — falling back to MODE-B (multipart)")

            # ══════════════════════════════════════════════════════════════════
            # MODE B — multipart/form-data (old/currently deployed Lambda)
            # ══════════════════════════════════════════════════════════════════
            if _needs_multipart or step1_resp.status_code not in (200, 201):
                print(f"  → [MODE-B] multipart POST for {filename}")
                with open(file_path, 'rb') as f:
                    files = {"pdfFile": (filename, f, "application/pdf")}
                    form_data = {
                        "doctorName":          doctor_name,
                        "patientName":         patient_name,
                        "reportType":          report_type,
                        "fileName":            filename,
                        "deviceId":            device_id_short,
                        "RhythmUltra_serial":  RhythmUltra_serial,
                    }
                    mp_resp = requests.post(
                        endpoint,
                        data=form_data,
                        files=files,
                        headers=api_headers,   # no Content-Type — requests sets multipart boundary
                        timeout=60,
                    )
                print(f"  ← [MODE-B] status={mp_resp.status_code}  body={mp_resp.text[:300]}")

                if mp_resp.status_code in (200, 201):
                    try:
                        mp_data = mp_resp.json()
                    except Exception:
                        mp_data = {}
                    # If we got a presigned URL back from the multipart Lambda, use it
                    mp_url = mp_data.get("uploadUrl")
                    if mp_url:
                        print("  → [MODE-B] PUTting PDF to presigned S3 URL…")
                        with open(file_path, 'rb') as f:
                            put2 = requests.put(mp_url, data=f,
                                                headers={"Content-Type": "application/pdf"},
                                                timeout=120)
                        if put2.status_code in (200, 204):
                            self._log_upload(file_path, {"status": "success"},
                                             {"type": "doctor_review", "doctor": doctor_name})
                            return {"status": "success", "message": "Report sent for review successfully"}
                        return {"status": "error", "message": f"S3 upload failed (HTTP {put2.status_code})"}

                    if mp_data.get("success") is True or mp_data.get("status") == "success":
                        print("[OK] Report sent via multipart flow")
                        self._log_upload(file_path, {"status": "success"},
                                         {"type": "doctor_review", "doctor": doctor_name})
                        return {"status": "success", "message": "Report sent for review successfully"}

                elif mp_resp.status_code == 404:
                    # ── REPORT NOT YET IN CLOUD DB ─────────────────────────────
                    # The Lambda needs the report in ecg_reports before assignment.
                    # Trigger an immediate cloud sync upload of this PDF, wait for
                    # ingestion, then retry once.
                    _err_body = {}
                    try:
                        _err_body = mp_resp.json()
                    except Exception:
                        pass
                    _is_not_found = (
                        "not found" in str(_err_body.get("message", "") or "").lower()
                        or "not found" in str(_err_body.get("error", {}).get("message", "")).lower()
                        or "old report" in mp_resp.text.lower()
                        or "report_not_found" in mp_resp.text.lower()
                    )
                    if _is_not_found:
                        print("  → Report not yet in cloud DB — triggering immediate sync upload & retry...")
                        try:
                            _synced = self.upload_complete_report_package(
                                pdf_path=file_path,
                                patient_data={"patient_name": patient_name, "name": patient_name},
                                ecg_data_file=None,
                                report_metadata=metadata or {},
                                report_type=report_type,
                            )
                            print(f"  → Sync upload result: {_synced}")
                            import time as _t
                            _t.sleep(3)  # give DB ingestion a moment
                        except Exception as _sync_err:
                            print(f"  → Sync upload failed: {_sync_err}")
                        # Retry MODE-B once after sync
                        print(f"  → [MODE-B RETRY] multipart POST for {filename}")
                        with open(file_path, 'rb') as f:
                            files2 = {"pdfFile": (filename, f, "application/pdf")}
                            mp_resp2 = requests.post(
                                endpoint,
                                data=form_data,
                                files=files2,
                                headers=api_headers,
                                timeout=60,
                            )
                        print(f"  ← [MODE-B RETRY] status={mp_resp2.status_code}  body={mp_resp2.text[:300]}")
                        if mp_resp2.status_code in (200, 201):
                            try:
                                mp_data2 = mp_resp2.json()
                            except Exception:
                                mp_data2 = {}
                            mp_url2 = mp_data2.get("uploadUrl")
                            if mp_url2:
                                print("  → [MODE-B RETRY] PUTting PDF to presigned S3 URL…")
                                with open(file_path, 'rb') as f:
                                    put3 = requests.put(mp_url2, data=f,
                                                        headers={"Content-Type": "application/pdf"},
                                                        timeout=120)
                                if put3.status_code in (200, 204):
                                    self._log_upload(file_path, {"status": "success"},
                                                     {"type": "doctor_review", "doctor": doctor_name})
                                    return {"status": "success", "message": "Report sent for review successfully"}
                            if mp_data2.get("success") is True or mp_data2.get("status") == "success":
                                print("[OK] Report sent via multipart retry")
                                self._log_upload(file_path, {"status": "success"},
                                                 {"type": "doctor_review", "doctor": doctor_name})
                                return {"status": "success", "message": "Report sent for review successfully"}
                        mp_resp = mp_resp2  # use retry response for final error

                # Handle HTTP 409 (Report already assigned to this doctor)
                if mp_resp.status_code == 409:
                    return {"status": "success", "message": "Report has already been sent to this doctor for review."}

                # Both modes failed — return the most recent error
                err_text = mp_resp.text if _needs_multipart else step1_resp.text
                err_code = mp_resp.status_code if _needs_multipart else step1_resp.status_code
                return {"status": "error", "message": f"Upload failed ({err_code}): {err_text[:200]}"}


            # Neither mode returned a success
            return {
                "status": "error",
                "message": f"Upload failed ({step1_resp.status_code}): {step1_resp.text[:200]}"
            }

        except Exception as e:
            print(f"[ERR] Error sending for review: {e}")
            return {"status": "error", "message": str(e)}





# Global instance
_cloud_uploader = None

def get_cloud_uploader():
    """Get or create global cloud uploader instance"""
    global _cloud_uploader
    if _cloud_uploader is None:
        _cloud_uploader = CloudUploader()
    return _cloud_uploader
