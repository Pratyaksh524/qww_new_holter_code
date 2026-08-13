import os


def _dev_autologin_enabled() -> bool:
    """
    True only when the developer auto-login bypass is explicitly switched on.

    The bypass skips the password prompt AND the licence gate (heartbeat,
    revocation, offline grace), so it defaults to OFF and is never enabled in a
    distributed build — CARDIOX_DEV_AUTOLOGIN is stripped from the installer's
    .env by build_exe.py.
    """
    return os.getenv("CARDIOX_DEV_AUTOLOGIN", "").strip().lower() in ("1", "true", "yes", "on")
from utils.app_paths import data_file
import json
import base64
import hashlib
import hmac
import secrets
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PyQt5.QtWidgets import (
    QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox, QStackedWidget, QWidget, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


def get_asset_path(asset_name):
    """
    Get the absolute path to an asset file in a portable way.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(os.path.dirname(os.path.dirname(script_dir)), "assets"),
        os.path.join(script_dir, "assets"),
        os.path.join(os.path.dirname(script_dir), "assets"),
        os.path.join(script_dir, "..", "assets"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return os.path.join(path, asset_name)
    
    # Fallback
    return os.path.join(script_dir, "..", "assets", asset_name)


USER_DATA_FILE = str(data_file("users.json"))
LEGACY_USER_DATA_FILE = str(Path(__file__).resolve().parents[1] / "users.json")


class SignIn:
    def __init__(self):
        self.users: Dict[str, Dict[str, Any]] = self.load_users()

    def _password_needs_hashing(self, stored: Any) -> bool:
        text = str(stored or "")
        return not text.startswith("pbkdf2_sha256$")

    def _hash_password(self, password: str, salt: Optional[str] = None) -> str:
        salt_bytes = (
            base64.urlsafe_b64decode(salt.encode("ascii"))
            if salt
            else secrets.token_bytes(16)
        )
        iterations = 260000
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, iterations)
        return "pbkdf2_sha256${}${}${}".format(
            iterations,
            base64.urlsafe_b64encode(salt_bytes).decode("ascii"),
            base64.urlsafe_b64encode(digest).decode("ascii"),
        )

    def _verify_password(self, password: str, stored: str) -> bool:
        text = str(stored or "")
        if text.startswith("pbkdf2_sha256$"):
            try:
                _, iterations, salt_b64, hash_b64 = text.split("$", 3)
                iterations_i = int(iterations)
                salt = base64.urlsafe_b64decode(salt_b64.encode("ascii"))
                expected = base64.urlsafe_b64decode(hash_b64.encode("ascii"))
                actual = hashlib.pbkdf2_hmac("sha256", str(password).encode("utf-8"), salt, iterations_i)
                return hmac.compare_digest(actual, expected)
            except Exception:
                return False
        return hmac.compare_digest(str(password), text)

    def _upgrade_password_if_needed(self, username: str, plain_password: str) -> None:
        record = self.users.get(username)
        if not isinstance(record, dict):
            return
        current = str(record.get("password", ""))
        if self._password_needs_hashing(current):
            record["password"] = self._hash_password(plain_password)
            self.users[username] = record
            self.save_users()

    def _migrate_legacy_format(self, raw: Any) -> Dict[str, Dict[str, Any]]:
        # Legacy format: {username: password}
        if isinstance(raw, dict):
            sample_values = list(raw.values())
            if len(sample_values) == 0:
                return {}
            if isinstance(sample_values[0], str):
                return {u: {"password": p} for u, p in raw.items()}
            # Already in new structured format
            return raw
        # Unknown/invalid -> start fresh
        return {}

    def _load_json_file(self, path: str) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self._migrate_legacy_format(data)
        except Exception:
            return {}

    def _merge_user_maps(self, primary: Dict[str, Dict[str, Any]], secondary: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {k: dict(v) for k, v in primary.items() if isinstance(v, dict)}
        for username, record in secondary.items():
            if not isinstance(record, dict):
                continue
            if username not in merged:
                merged[username] = dict(record)
                continue
            for key, value in record.items():
                if key not in merged[username] or merged[username].get(key) in ("", None):
                    merged[username][key] = value
        return merged

    def load_users(self) -> Dict[str, Dict[str, Any]]:
        canonical = self._load_json_file(USER_DATA_FILE)
        legacy = {}
        if os.path.abspath(LEGACY_USER_DATA_FILE) != os.path.abspath(USER_DATA_FILE):
            legacy = self._load_json_file(LEGACY_USER_DATA_FILE)
        merged = self._merge_user_maps(canonical, legacy)
        if merged != canonical and merged:
            try:
                self.users = merged
                self.save_users()
            except Exception:
                pass
        return merged

    def save_users(self) -> None:
        os.makedirs(os.path.dirname(USER_DATA_FILE), exist_ok=True)
        with open(USER_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.users, f, indent=2)

    @staticmethod
    def _normalize_phone(value: str) -> str:
        return "".join(ch for ch in str(value or "") if ch.isdigit())

    @staticmethod
    def _normalize_name(value: str) -> str:
        return " ".join(str(value or "").strip().lower().split())

    def _identifier_matches_record(self, identifier: str, uname: str, record: Dict[str, Any]) -> bool:
        ident = str(identifier).strip()
        if not ident:
            return False
        ident_norm = ident.lower()
        ident_name = self._normalize_name(ident)
        ident_phone = self._normalize_phone(ident)

        candidates = [
            uname,
            str(record.get("login_id", "")),
            str(record.get("login_username", "")),
            str(record.get("login_identifier", "")),
            str(record.get("canonical_username", "")),
            str(record.get("username", "")),
            str(record.get("phone", "")),
            str(record.get("master_phone", "")),
            str(record.get("contact", "")),
            str(record.get("full_name", "")),
        ]
        aliases = record.get("login_aliases", [])
        if isinstance(aliases, (list, tuple)):
            candidates.extend(str(alias or "") for alias in aliases)
        for candidate in candidates:
            text = str(candidate or "").strip()
            if not text:
                continue
            if ident == text or ident_norm == text.lower():
                return True
            if ident_name and ident_name == self._normalize_name(text):
                return True
            cand_phone = self._normalize_phone(text)
            if ident_phone and cand_phone and ident_phone == cand_phone:
                return True
            # Let users sign in with the first word of their full name too.
            # This keeps the offline JSON login usable when the signup name is
            # entered as a longer legal name but the user remembers the short form.
            full_name_parts = self._normalize_name(str(record.get("full_name", ""))).split()
            if ident_name and full_name_parts and ident_name == full_name_parts[0]:
                return True
        return False

    def _find_user_record(self, identifier: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        # Identifier may be Login ID (phone), username dict key, or full name
        ident = str(identifier).strip()
        if ident in self.users:
            return ident, self.users[ident]
        for uname, record in self.users.items():
            if not isinstance(record, dict):
                continue
            if self._identifier_matches_record(ident, uname, record):
                return uname, record
        return None

    def sign_in_user(self, username: str, password: str) -> bool:
        # Backward-compatible: validate using password or fallback serial
        return self.validate_credentials(username, password)

    def sign_in_user_allow_serial(self, identifier: str, secret: str) -> bool:
        return self.validate_credentials(identifier, secret)

    def _remove_local_user(self, username: str) -> None:
        """Delete local cached account + license token if backend confirms user is revoked/deleted."""
        found = self._find_user_record(username)
        if not found:
            self._clear_license_files()
            return
        found_username, _ = found
        if found_username in self.users:
            del self.users[found_username]
            self.save_users()
            print(f"\U0001f5d1\ufe0f Removed local user cache for '{found_username}' (account revoked or deleted on backend).")
        self._clear_license_files()

    def _clear_license_files(self) -> None:
        """Clear local license token (cardiox.lic) and stored license key on revocation."""
        try:
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from utils.license_manager import clear_stored_key, clear_license_cache
            clear_stored_key()
            clear_license_cache()
            print("\U0001f5d1\ufe0f License token and stored key cleared from system (account revoked).")
        except Exception as e:
            print(f"\u26a0\ufe0f Could not clear license files: {e}")

    def _check_license_server(self, username: str, password: str) -> str:
        """
        Query backend license server to validate credentials & license state.
        Returns: 'ACTIVE', 'REVOKED', 'INVALID', or 'OFFLINE'
        """
        try:
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from utils.license_manager import validate_with_credentials

            # validate_with_credentials() sends the same password form
            # register_device() used to create the seat (SHA-256), retrying with
            # the raw password for seats created by older builds. Posting the raw
            # password only made the server's bcrypt comparison fail and dropped
            # every online login into the local fallback.
            data = validate_with_credentials(username, password)
        except Exception as net_err:
            print(f"\u26a0\ufe0f License server offline or unreachable: {net_err}")
            return "OFFLINE"

        print(f"[License DEBUG] validate response: {str(data)[:300]}")

        if data.get("offline"):
            return "OFFLINE"

        if data.get("success") is True or data.get("valid") is True or data.get("status") == "active":
            return "ACTIVE"

        err = str(data.get("error", "")).upper()
        msg = str(data.get("message", "")).upper()
        combined = f"{err} {msg}"

        # Only an explicit revocation may purge the local account and licence.
        # Seat-state answers (SEAT_INACTIVE / SEAT_RELEASED / seat not found)
        # mean the cached token is stale, not that the user lost their licence \u2014
        # deleting their account over one would lock out a paying customer whose
        # only problem is an out-of-date token.
        revoked_codes = [
            "USER_DELETED", "ACCOUNT_REVOKED", "LICENSE_REVOKED",
            "LICENSE EXPIRED", "LICENSE_EXPIRED", "REVOKED",
        ]
        if any(code in combined for code in revoked_codes):
            return "REVOKED"

        return "INVALID"

    def validate_credentials(self, username: str, password: str) -> bool:
        """
        Online-first authentication with local JSON fallback.

        Flow:
          ACTIVE  -> server confirmed active -> allow & update timestamp
          REVOKED -> server explicitly revoked/deleted -> purge local cache -> block
          INVALID -> server returned 401 (bcrypt hash mismatch) -> fall back to local
          OFFLINE -> server unreachable -> fall back to local + 7-day grace window
        """
        import time

        server_status = self._check_license_server(username, password)

        if server_status == "ACTIVE":
            found = self._find_user_record(username)
            if found:
                found_username, record = found
                record["last_server_validation"] = time.time()
                self.users[found_username] = record
                self.save_users()
            return True

        if server_status == "REVOKED":
            self._remove_local_user(username)
            return False

        if server_status == "INVALID":
            print(f"\u26a0\ufe0f Server returned INVALID for '{username}'. Falling back to local check.")
            return self._validate_local_credentials(username, password)

        if server_status == "OFFLINE":
            found = self._find_user_record(username)
            if not found:
                return False
            found_username, record = found
            last_validated = record.get("last_server_validation", 0)
            offline_grace_hours = 168.0  # 7 days
            if last_validated > 0:
                hours_since = (time.time() - last_validated) / 3600.0
                if hours_since > offline_grace_hours:
                    print(f"\u26a0\ufe0f Offline login rejected: {hours_since:.1f} hrs ago (limit: 168h / 7 days).")
                    return False
            return self._validate_local_credentials(username, password)

        return False

    def _validate_local_credentials(self, username: str, password: str) -> bool:
        """Validate credentials locally against stored PBKDF2 hash in users.json."""
        found = self._find_user_record(username)
        if not found:
            return False
        found_username, record = found
        if not isinstance(record, dict):
            return False
        stored_password = str(record.get("password", ""))
        if not stored_password:
            return False
        if self._verify_password(password, stored_password):
            if self._password_needs_hashing(stored_password):
                self._upgrade_password_if_needed(found_username, str(password))
            return True
        return False

    def _is_unique(self, key: str, value: str) -> bool:
        if not value:
            return True
        for uname, rec in self.users.items():
            if str(rec.get(key, "")) == str(value):
                return False
        return True

    def register_user_with_details(
        self,
        username: str,
        password: str,
        full_name: Optional[str] = None,
        phone: Optional[str] = None,
        serial_id: Optional[str] = None,
        email: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        # If username already exists, allow update (overwrite) instead of blocking
        is_update = username in self.users
        if not is_update:
            # NOTE: serial_id and phone uniqueness are enforced on the server side
            # (via hardware fingerprint + RhythmUltra serial checks). We do NOT
            # enforce them locally because the same setup installer (including
            # users.json) may be installed on multiple machines — a local
            # duplicate-serial check would incorrectly block the 2nd machine.
            if full_name and not self._is_unique("full_name", full_name):
                return False, "Full Name already registered on this machine."

        from datetime import datetime
        login_id = str(phone or username).strip()

        # NOTE: this method saves the LOCAL account only. It must never POST to
        # /register \u2014 the seat is created once by license_manager.register_device()
        # (main.py's RegisterWorker), and that call's token is the one written to
        # cardiox.lic. A second /register for the same fingerprint makes the server
        # issue a new seat and deactivate the first, so the token on disk points at
        # a dead seat and every later heartbeat answers SEAT_INACTIVE \u2014 which the
        # login gate reported as "device seat was not found" and offered to fix by
        # re-registering, creating another seat and repeating the loop.
        record: Dict[str, Any] = {
            "password": self._hash_password(password),
            "full_name": full_name or "",
            "phone": phone or login_id,
            "login_id": login_id,
            "login_username": login_id,
            "login_identifier": login_id,
            "canonical_username": login_id,
            "username": login_id,
            "serial_id": serial_id or "",
            "email": email or "",
            "signup_date": datetime.now().strftime("%Y-%m-%d"),
        }
        login_aliases = []
        if full_name:
            login_aliases.append(full_name)
            short_name = str(full_name).strip().split()[0] if str(full_name).strip() else ""
            if short_name and short_name.lower() != str(full_name).strip().lower():
                login_aliases.append(short_name)
        if phone:
            login_aliases.append(phone)
        if login_aliases:
            record["login_aliases"] = login_aliases
        if isinstance(extra, dict):
            for k, v in extra.items():
                if k not in record:
                    record[k] = v
        self.users[username] = record
        self.save_users()
        
        # ========================================
        # AUTOMATIC CLOUD UPLOAD - USER SIGNUP
        # ========================================
        # Upload user signup data to cloud automatically
        try:
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from utils.cloud_uploader import get_cloud_uploader
            from datetime import datetime
            
            cloud_uploader = get_cloud_uploader()
            if cloud_uploader.is_configured():
                # Prepare user data for cloud upload
                user_data = {
                    "username": username,
                    "full_name": full_name or "",
                    "phone": phone or "",
                    "serial_id": serial_id or "",
                    "email": email or "",
                    "registration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                # Add extra fields if present
                if isinstance(extra, dict):
                    for k, v in extra.items():
                        if k not in user_data:
                            user_data[k] = str(v) if v is not None else ""
                
                # Upload to cloud
                result = cloud_uploader.upload_user_signup(user_data)
                if result.get('status') == 'success':
                    print(f" User signup automatically uploaded to cloud")
                else:
                    print(f"  Cloud upload failed: {result.get('message', 'Unknown error')}")
            else:
                print("  Cloud not configured - user saved locally only")
        except Exception as e:
            print(f"  Cloud upload error: {e}")
            # Don't fail registration if cloud upload fails
        
        return True, "Registration successful."

    def register_user(self, username: str, password: str) -> bool:
        # Maintain legacy API, without extra details
        ok, _ = self.register_user_with_details(username=username, password=password)
        return ok


class LoginRegisterDialog(QDialog):
    def __init__(self):
        super().__init__()
        
        # Set responsive size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(600, 400)  # Minimum size for usability
        
        self.setWindowTitle("CardioX - Sign In / Sign Up")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self.setStyleSheet("""
            QDialog { background: #fff; border-radius: 18px; }
            QLabel { font-size: 15px; color: #222; }
            QLineEdit { border: 2px solid #ff6600; border-radius: 8px; padding: 6px 10px; font-size: 15px; background: #f7f7f7; }
            QPushButton { background: #ff6600; color: white; border-radius: 10px; padding: 8px 0; font-size: 16px; font-weight: bold; }
            QPushButton:hover { background: #ff8800; }
        """)
        self.sign_in_logic = SignIn()
        self.init_ui()
        self.result = False
        self.username = None
        self.user_details = {}

    def exec_(self):
        # Development-only auto-login for the 'cardiomac' test account.
        # Returns QDialog.Accepted with no password check and no licence
        # verification, so it is gated OFF by default. See _dev_autologin_enabled().
        if _dev_autologin_enabled():
            try:
                print("🔑 [DEV] Bypassing login screen for user: cardiomac (9560350477)")
                found = self.sign_in_logic._find_user_record("cardiomac")
                if found:
                    username, record = found
                    self.result = True
                    self.username = username
                    self.user_details = record
                    return QDialog.Accepted
                else:
                    print("⚠️ User 'cardiomac' not found in users.json! Showing login dialog...")
            except Exception as e:
                print(f"⚠️ Bypass login error: {e}")
        return super().exec_()

    def init_ui(self):
        from PyQt5.QtWidgets import QSizePolicy
        from PyQt5.QtGui import QMovie, QPixmap
        
        main_layout = QHBoxLayout()
        
        # Left: Logo/Image with plasma effect background
        logo_widget = QWidget()
        logo_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        logo_layout = QVBoxLayout()
        logo_layout.setAlignment(Qt.AlignCenter)
        
        # Clean background instead of plasma effect
        bg_label = QLabel()
        bg_label.setFixedSize(260, 260)
        bg_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        bg_label.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                stop:0 #667eea, stop:1 #764ba2);
            border-radius: 130px;
        """)
        
        # ECG image on top of clean background
        logo_label = QLabel()
        logo_path = get_asset_path('ECG1.png')
        if os.path.exists(logo_path):
            logo_pixmap = QPixmap(logo_path).scaled(180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(logo_pixmap)
            logo_label.setStyleSheet("background: transparent;")
            logo_label.setAlignment(Qt.AlignCenter)
            logo_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        else:
            logo_label.setText("<b>PulseMonitor</b>")
            logo_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Stack background and ECG image
        bg_container = QVBoxLayout()
        bg_container.setAlignment(Qt.AlignCenter)
        bg_container.addWidget(bg_label, alignment=Qt.AlignCenter)
        bg_container.addWidget(logo_label, alignment=Qt.AlignCenter)
        
        logo_layout.addStretch(1)
        logo_layout.addLayout(bg_container)
        logo_layout.addStretch(1)
        logo_widget.setLayout(logo_layout)
        
        # Right: Form
        form_widget = QWidget()
        form_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        form_layout = QVBoxLayout()
        form_layout.setAlignment(Qt.AlignCenter)
        self.stacked = QStackedWidget(self)
        self.stacked.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.login_widget = self.create_login_widget()
        self.register_widget = self.create_register_widget()
        self.stacked.addWidget(self.login_widget)
        self.stacked.addWidget(self.register_widget)
        
        btn_layout = QHBoxLayout()
        self.login_tab = QPushButton("Sign In")
        self.signup_tab = QPushButton("Sign Up")
        
        # Make buttons responsive
        for btn in [self.login_tab, self.signup_tab]:
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            btn.setMinimumWidth(100)
        
        self.login_tab.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        self.signup_tab.clicked.connect(lambda: self.stacked.setCurrentIndex(1))
        btn_layout.addWidget(self.login_tab)
        btn_layout.addWidget(self.signup_tab)
        
        title = QLabel("<span style='font-family:cursive;font-size:32px;color:#222;'>PulseMonitor</span>")
        title.setAlignment(Qt.AlignCenter)
        title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        form_layout.addWidget(title)
        form_layout.addLayout(btn_layout)
        form_layout.addWidget(self.stacked)
        form_widget.setLayout(form_layout)
        
        # Add to main layout (image left, form right) with responsive proportions
        main_layout.addWidget(logo_widget, 2)  # Logo takes 2 parts
        main_layout.addWidget(form_widget, 3)  # Form takes 3 parts (more space)
        self.setLayout(main_layout)

    def create_login_widget(self):
        widget = QWidget()
        widget.setStyleSheet("background: transparent;")
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        self.login_username = QLineEdit()
        self.login_username.setPlaceholderText("Username")
        self.login_username.setStyleSheet("""
            QLineEdit {
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 10px;
                padding: 12px 15px;
                font-size: 16px;
                color: white;
                font-weight: bold;
            }
            QLineEdit::placeholder {
                color: rgba(255, 255, 255, 0.7);
            }
            QLineEdit:focus {
                border: 2px solid rgba(255, 255, 255, 0.6);
                background: rgba(255, 255, 255, 0.25);
            }
        """)
        
        self.login_password = QLineEdit()
        self.login_password.setPlaceholderText("Password")
        self.login_password.setEchoMode(QLineEdit.Password)
        self.login_password.setStyleSheet(self.login_username.styleSheet())
        
        login_btn = QPushButton("Sign In")
        login_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.25);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.4);
                border-radius: 12px;
                padding: 12px 0;
                font-size: 18px;
                font-weight: bold;
                margin-top: 10px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.35);
                border: 1px solid rgba(255, 255, 255, 0.6);
            }
            QPushButton:pressed {
                background: rgba(255, 255, 255, 0.45);
            }
        """)
        login_btn.clicked.connect(self.handle_login)
        
        layout.addWidget(self.login_username)
        layout.addWidget(self.login_password)
        layout.addWidget(login_btn)
        layout.addStretch(1)
        widget.setLayout(layout)
        return widget

    def create_register_widget(self):
        widget = QWidget()
        widget.setStyleSheet("background: transparent;")
        layout = QVBoxLayout()
        layout.setSpacing(12)
        
        # Create all input fields with glass morphism style
        self.reg_username = QLineEdit()
        self.reg_username.setPlaceholderText("Username (will be your phone)")
        self.reg_username.setStyleSheet(self.login_username.styleSheet())
        
        self.reg_serial = QLineEdit()
        self.reg_serial.setPlaceholderText("Machine Serial ID")
        self.reg_serial.setReadOnly(True)
        self.reg_serial.setStyleSheet(self.login_username.styleSheet())

        self.reg_password = QLineEdit()
        self.reg_password.setPlaceholderText("Password")
        self.reg_password.setEchoMode(QLineEdit.Password)
        self.reg_password.setStyleSheet(self.login_username.styleSheet())
        
        self.reg_confirm = QLineEdit()
        self.reg_confirm.setPlaceholderText("Confirm Password")
        self.reg_confirm.setEchoMode(QLineEdit.Password)
        self.reg_confirm.setStyleSheet(self.login_username.styleSheet())
        self.reg_confirm.returnPressed.connect(self.handle_register)
        
        self.reg_fullname = QLineEdit()
        self.reg_fullname.setPlaceholderText("Full Name")
        self.reg_fullname.setStyleSheet(self.login_username.styleSheet())
        
        self.reg_age = QLineEdit()
        self.reg_age.setPlaceholderText("Age")
        self.reg_age.setStyleSheet(self.login_username.styleSheet())
        
        self.reg_gender = QLineEdit()
        self.reg_gender.setPlaceholderText("Gender")
        self.reg_gender.setStyleSheet(self.login_username.styleSheet())
        
        self.reg_contact = QLineEdit()
        self.reg_contact.setPlaceholderText("Contact Number")
        self.reg_contact.setStyleSheet(self.login_username.styleSheet())
        
        self.reg_email = QLineEdit()
        self.reg_email.setPlaceholderText("Email Address")
        self.reg_email.setStyleSheet(self.login_username.styleSheet())
        
        register_btn = QPushButton("Sign Up")
        register_btn.setStyleSheet(self.login_btn.styleSheet())
        register_btn.clicked.connect(self.handle_register)
        
        # Set size policy for all widgets
        for w in [self.reg_username, self.reg_serial, self.reg_password, self.reg_confirm, self.reg_fullname, 
                  self.reg_age, self.reg_gender, self.reg_contact, self.reg_email, register_btn]:
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Add widgets to layout
        layout.addWidget(self.reg_username)
        layout.addWidget(self.reg_serial)
        layout.addWidget(self.reg_password)
        layout.addWidget(self.reg_confirm)
        layout.addWidget(self.reg_fullname)
        layout.addWidget(self.reg_age)
        layout.addWidget(self.reg_gender)
        layout.addWidget(self.reg_contact)
        layout.addWidget(self.reg_email)
        layout.addWidget(register_btn)
        layout.addStretch(1)
        widget.setLayout(layout)
        return widget

    def handle_login(self):
        username = self.login_username.text()
        password_or_serial = self.login_password.text()
        if self.sign_in_logic.sign_in_user_allow_serial(username, password_or_serial):
            self.result = True
            self.username = username
            self.accept()
        else:
            QMessageBox.warning(self, "Error", "Invalid username or password/serial.")

    def handle_register(self):
        username = self.reg_username.text()
        password = self.reg_password.text()
        confirm = self.reg_confirm.text()
        serial_id = self.reg_serial.text().strip()
        if serial_id in ("RhythmUltra device connection lost. Please reconnect the device", ""):
            serial_id = ""
        fullname = self.reg_fullname.text()
        age = self.reg_age.text()
        gender = self.reg_gender.text()
        contact = self.reg_contact.text()
        email = self.reg_email.text()
        if not username or not password:
            QMessageBox.warning(self, "Error", "Username and password are required.")
            return
        if password != confirm:
            QMessageBox.warning(self, "Error", "Passwords do not match.")
            return
        if not fullname or not age or not gender or not contact or not email:
            QMessageBox.warning(self, "Error", "All details are required.")
            return
        ok, msg = self.sign_in_logic.register_user_with_details(
            username=username,
            password=password,
            full_name=fullname,
            phone=contact,
            serial_id=serial_id,
            email=email,
            extra={"age": age, "gender": gender}
        )
        if not ok:
            QMessageBox.warning(self, "Error", msg)
            return
        QMessageBox.information(self, "Success", "Registration successful! You can now sign in.")
        self.stacked.setCurrentIndex(0)
