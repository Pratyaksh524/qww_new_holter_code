"""
Organization Management Module

This module handles organization creation, storage, and role selection functionality
for the CardioX application.
"""

import os
from utils.app_paths import data_file
import json
from datetime import datetime
from PyQt5.QtWidgets import (
    QDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, 
    QMessageBox, QInputDialog, QLineEdit, QWidget, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, QRegularExpression
from PyQt5.QtGui import QFont, QPixmap, QIntValidator, QRegularExpressionValidator
from config.settings import resource_path


class BaseDialogMixin:
    """Mixin class to handle common dialog close behavior"""
    def closeEvent(self, event):
        """Handle close event - return to role selection dialog"""
        # Instead of closing, reject this dialog and return to role selection
        self.reject()
        # The parent RoleSelectionDialog will handle showing the role selection again
        event.ignore()


# File paths for different user types based on organization
DOCTOR_HEAD_USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doctor_head_users.json")
HCP_HEAD_USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hcp_head_users.json")
ORGANIZATIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "organizations.json")

def get_created_users_file(creator_full_name, creator_role):
    """Get the JSON file path for users created by a specific Doctor Head/HCP Head"""
    # Sanitize the creator name to create a valid filename
    safe_name = creator_full_name.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
    filename = f"users_created_by_{safe_name}_{creator_role.lower().replace(' ', '_')}.json"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

def load_created_users(creator_full_name, creator_role):
    """Load users created by a specific Doctor Head/HCP Head"""
    try:
        users_file = get_created_users_file(creator_full_name, creator_role)
        if os.path.exists(users_file):
            with open(users_file, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading created users for {creator_full_name}: {e}")
        return {}

def save_created_users(users, creator_full_name, creator_role):
    """Save users created by a specific Doctor Head/HCP Head"""
    try:
        users_file = get_created_users_file(creator_full_name, creator_role)
        with open(users_file, "w") as f:
            json.dump(users, f, indent=2)
        print(f"Saved {len(users)} users created by {creator_full_name} ({creator_role}) to {users_file}")
        return True
    except Exception as e:
        print(f"Error saving created users: {e}")
        return False

def get_all_created_users_count(creator_full_name, creator_role, organization=None):
    """Get count of users created by a specific Doctor Head/HCP Head, optionally filtered by organization"""
    try:
        created_users = load_created_users(creator_full_name, creator_role)
        count = 0
        
        if organization:
            # Count users in specific organization
            for username, user_data in created_users.items():
                if user_data.get('organization') == organization:
                    count += 1
        else:
            # Count all users
            count = len(created_users)
        
        return count
    except Exception:
        return 0

def cleanup_orphaned_users():
    """Remove users whose organizations no longer exist"""
    try:
        # Load current organizations
        if os.path.exists(ORGANIZATIONS_FILE):
            with open(ORGANIZATIONS_FILE, "r") as f:
                organizations = json.load(f)
        else:
            organizations = {}
        
        print(f"DEBUG: Current organizations: {list(organizations.keys())}")
        
        # Clean Doctor Head users
        doctor_head_users = load_doctor_head_users()
        orphaned_users = []
        for username, user_data in doctor_head_users.items():
            user_org = user_data.get('organization')
            if user_org not in organizations:
                orphaned_users.append(username)
                print(f"DEBUG: Found orphaned Doctor Head user {username} with non-existent org {user_org}")
        
        for username in orphaned_users:
            doctor_head_users.pop(username, None)
            print(f"DEBUG: Removed orphaned Doctor Head user {username}")
        
        if orphaned_users:
            save_head_users(doctor_head_users, 'Doctor Head')
            print(f"DEBUG: Cleaned up {len(orphaned_users)} orphaned Doctor Head users")
        
        # Clean HCP Head users
        hcp_head_users = load_hcp_head_users()
        orphaned_users = []
        for username, user_data in hcp_head_users.items():
            user_org = user_data.get('organization')
            if user_org not in organizations:
                orphaned_users.append(username)
                print(f"DEBUG: Found orphaned HCP Head user {username} with non-existent org {user_org}")
        
        for username in orphaned_users:
            hcp_head_users.pop(username, None)
            print(f"DEBUG: Removed orphaned HCP Head user {username}")
        
        if orphaned_users:
            save_head_users(hcp_head_users, 'HCP Head')
            print(f"DEBUG: Cleaned up {len(orphaned_users)} orphaned HCP Head users")
        
        # Clean creator-specific user files for orphaned organizations
        import glob
        src_dir = os.path.dirname(os.path.abspath(__file__))
        creator_user_files = glob.glob(os.path.join(src_dir, "users_created_by_*.json"))
        
        for file_path in creator_user_files:
            try:
                with open(file_path, "r") as f:
                    created_users = json.load(f)
                
                orphaned_created_users = []
                for username, user_data in created_users.items():
                    user_org = user_data.get('organization')
                    if user_org not in organizations:
                        orphaned_created_users.append(username)
                        print(f"DEBUG: Found orphaned created user {username} in {os.path.basename(file_path)} with non-existent org {user_org}")
                
                if orphaned_created_users:
                    for username in orphaned_created_users:
                        created_users.pop(username, None)
                        print(f"DEBUG: Removed orphaned created user {username} from {os.path.basename(file_path)}")
                    
                    # Save updated file
                    with open(file_path, "w") as f:
                        json.dump(created_users, f, indent=2)
                    print(f"DEBUG: Cleaned up {len(orphaned_created_users)} orphaned users from {os.path.basename(file_path)}")
                    
                    # Remove empty files
                    if not created_users:
                        os.remove(file_path)
                        print(f"DEBUG: Removed empty creator file: {os.path.basename(file_path)}")
                        
            except Exception as e:
                print(f"DEBUG: Error cleaning up creator file {os.path.basename(file_path)}: {e}")
            
    except Exception as e:
        print(f"DEBUG: Error cleaning up orphaned users: {e}")


def load_head_users():
    """Load head users from appropriate file based on role"""
    try:
        # Load Doctor Head users
        if self.role == 'Doctor Head':
            return load_doctor_head_users()
        elif self.role == 'HCP Head':
            return load_hcp_head_users()
        else:
            return {}
    except Exception as e:
        print(f"Error loading head users: {e}")
        return {}

def save_head_users(head_users, role):
    """Save head users to appropriate file based on role"""
    try:
        if role == 'Doctor Head':
            with open(DOCTOR_HEAD_USERS_FILE, "w") as f:
                json.dump(head_users, f, indent=2)
            print(f"Saved {len(head_users)} Doctor Head users to {DOCTOR_HEAD_USERS_FILE}")
        elif role == 'HCP Head':
            with open(HCP_HEAD_USERS_FILE, "w") as f:
                json.dump(head_users, f, indent=2)
            print(f"Saved {len(head_users)} HCP Head users to {HCP_HEAD_USERS_FILE}")
        return True
    except Exception as e:
        print(f"Error saving head users: {e}")
        return False

def save_user_to_main_users(user_data, username):
    """Save user to main users.json file for normal login"""
    try:
        # Save to the root users.json file (where sign_in.py looks for it)
        users_file = str(data_file("users.json"))
        
        # Load existing users
        users = {}
        if os.path.exists(users_file):
            with open(users_file, "r") as f:
                users = json.load(f)
        
        # Add/update user in main users file
        users[username] = user_data
        
        # Save to main users file
        with open(users_file, "w") as f:
            json.dump(users, f, indent=2)
        
        print(f"Saved user {username} to main users.json file")
        return True
    except Exception as e:
        print(f"Error saving user to main users.json: {e}")
        return False

def load_main_users():
    """Load the root users.json (shared login store)."""
    try:
        users_file = str(data_file("users.json"))
        if os.path.exists(users_file):
            with open(users_file, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading main users.json: {e}")
        return {}

def load_doctor_head_users():
    """Load Doctor Head users from separate file"""
    try:
        if os.path.exists(DOCTOR_HEAD_USERS_FILE):
            with open(DOCTOR_HEAD_USERS_FILE, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading Doctor Head users: {e}")
        return {}

def load_hcp_head_users():
    """Load HCP Head users from separate file"""
    try:
        if os.path.exists(HCP_HEAD_USERS_FILE):
            with open(HCP_HEAD_USERS_FILE, "r") as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error loading HCP Head users: {e}")
        return {}

# Note: cleanup_orphaned_users() is now only called when needed (e.g., after organization deletion)
# Not running on module import to prevent premature cleanup


class OrganizationManager:
    """Handles organization data management"""
    
    def __init__(self):
        self.organizations_file = resource_path("organizations.json")
    
    def load_organizations(self):
        """Load existing organizations from file"""
        try:
            if os.path.exists(self.organizations_file):
                with open(self.organizations_file, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading organizations: {e}")
            return {}
    
    def save_organizations(self, organizations):
        """Save organizations to file"""
        try:
            print(f"DEBUG: Saving organizations to {self.organizations_file}")
            print(f"DEBUG: Organizations data: {organizations}")
            with open(self.organizations_file, "w") as f:
                json.dump(organizations, f, indent=2)
            print(f"DEBUG: File saved successfully")
            return True
        except Exception as e:
            print(f"Error saving organizations: {e}")
            return False
    
    def add_organization(self, org_name):
        """Add a new organization"""
        if not org_name or not org_name.strip():
            return False, "Organization name cannot be empty"
        
        organizations = self.load_organizations()
        org_name_clean = org_name.strip()
        
        # Check if organization already exists
        if org_name_clean in organizations:
            return False, "Organization already exists"
        
        # Add new organization
        organizations[org_name_clean] = {
            "name": org_name_clean,
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "active"
        }
        
        if self.save_organizations(organizations):
            return True, "Organization added successfully"
        else:
            return False, "Failed to save organization"


class RoleSelectionDialog(QDialog):
    """Dialog for selecting user role after organization creation"""
    
    def __init__(self, parent=None, organization_name="", is_existing_organization=False):
        super().__init__(parent.parent_dialog if parent else None)  # Pass the actual QWidget parent
        self.organization_name = organization_name
        self.is_existing_organization = is_existing_organization
        self.selected_role = None
        self.selected_organization = organization_name
        self.parent_handler = parent  # Store reference to parent handler
        self.init_ui()
    
    def closeEvent(self, event):
        """Handle close event - return to existing organization dialog"""
        if self.is_existing_organization and self.parent_handler:
            # Instead of closing, reject this dialog and return to existing org dialog
            self.reject()
            # Reopen the existing organization dialog
            self.parent_handler.handle_existing_organization_request()
        else:
            # For new organizations, just close normally
            super().closeEvent(event)
    
    def init_ui(self):
        """Initialize the role selection dialog UI"""
        self.setWindowTitle("Select Role")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setMinimumSize(400, 350 if self.is_existing_organization else 300)
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #1a1a2e, stop:1 #16213e);
                color: white;
            }
            QLabel {
                color: white;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton {
                background: #ff6600;
                color: white;
                border-radius: 10px;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: bold;
                min-height: 50px;
            }
            QPushButton:hover {
                background: #ff8800;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel(f"Organization: {self.organization_name}")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; margin-bottom: 20px;")
        layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Select your role:")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        
        layout.addStretch(1)
        
        # Initialize login flag
        self.is_login = False
        
        # Role buttons
        doctor_head_btn = QPushButton("Sign up as Doctor Head")
        hcp_head_btn = QPushButton("Sign up as HCP Head")
        
        # Store selected role and organization
        def select_doctor_head():
            self.selected_role = "Doctor Head"
            self.selected_organization = self.organization_name
            self.is_login = False
            # Show sign-up dialog instead of just accepting
            self.show_signup_dialog()
            
        def select_hcp_head():
            self.selected_role = "HCP Head"
            self.selected_organization = self.organization_name
            self.is_login = False
            # Show sign-up dialog instead of just accepting
            self.show_signup_dialog()
        
        doctor_head_btn.clicked.connect(select_doctor_head)
        hcp_head_btn.clicked.connect(select_hcp_head)
        
        layout.addWidget(doctor_head_btn)
        layout.addSpacing(10)
        layout.addWidget(hcp_head_btn)
        
        # Add login options for existing organizations
        if self.is_existing_organization:
            layout.addSpacing(20)
            
            # Add separator line
            separator = QLabel("─ or login as existing user ─")
            separator.setAlignment(Qt.AlignCenter)
            separator.setStyleSheet("color: #ff6600; font-size: 12px; margin: 10px 0;")
            layout.addWidget(separator)
            
            # Create login buttons with smaller size
            login_doctor_btn = QPushButton("Login as Doctor Head")
            login_hcp_btn = QPushButton("Login as HCP Head")
            
            # Style for login buttons (smaller)
            login_button_style = """
                QPushButton {
                    background: rgba(255, 102, 0, 0.8);
                    color: white;
                    border-radius: 8px;
                    padding: 8px 16px;
                    font-size: 14px;
                    font-weight: bold;
                    min-height: 35px;
                }
                QPushButton:hover {
                    background: rgba(255, 136, 0, 0.9);
                }
            """
            
            login_doctor_btn.setStyleSheet(login_button_style)
            login_hcp_btn.setStyleSheet(login_button_style)
            
            def login_as_doctor_head():
                self.selected_role = "Doctor Head"
                self.selected_organization = self.organization_name
                self.is_login = True  # Flag to indicate login vs signup
                # Show login dialog instead of just accepting
                self.show_login_dialog()
                
            def login_as_hcp_head():
                self.selected_role = "HCP Head"
                self.selected_organization = self.organization_name
                self.is_login = True  # Flag to indicate login vs signup
                # Show login dialog instead of just accepting
                self.show_login_dialog()
            
            login_doctor_btn.clicked.connect(login_as_doctor_head)
            login_hcp_btn.clicked.connect(login_as_hcp_head)
            
            layout.addWidget(login_doctor_btn)
            layout.addSpacing(8)
            layout.addWidget(login_hcp_btn)
        
        layout.addStretch(1)
    
    def get_selection(self):
        """Get the selected role and organization"""
        return self.selected_role, self.selected_organization, getattr(self, 'is_login', False)
    
    def show_signup_dialog(self):
        """Show the sign-up dialog for the selected role"""
        signup_dialog = SignUpDialog(self, self.selected_role, self.selected_organization)
        
        if signup_dialog.exec_() == QDialog.Accepted:
            # Get user data and open dashboard
            user_data = signup_dialog.get_user_data()
            print(f"DEBUG: Signup dialog returned Accepted. User data: {user_data}")
            if user_data:
                self.signup_user_data = user_data
                print(f"DEBUG: signup_user_data set to: {self.signup_user_data}")
                self.accept()  # Close the role selection dialog
                # Open dashboard will be handled by the calling code
            else:
                # Sign-up failed or was cancelled
                print("DEBUG: User data is None after signup")
                self.selected_role = None
                self.selected_organization = None
                self.reject()
        else:
            # Sign-up was cancelled via close button - return to role selection
            print("DEBUG: Signup dialog was cancelled or rejected - returning to role selection")
            # Don't set role/org to None, keep the role selection dialog open
            # User can choose a different option or close the main dialog
    
    def show_login_dialog(self):
        """Show the login dialog for the selected role"""
        login_dialog = LoginDialog(self, self.selected_role, self.selected_organization)
        
        if login_dialog.exec_() == QDialog.Accepted:
            # Get user data and open dashboard
            user_data = login_dialog.get_user_data()
            if user_data:
                self.login_user_data = user_data
                self.accept()  # Close the role selection dialog
                # Open dashboard will be handled by the calling code
            else:
                # Login failed or was cancelled
                self.selected_role = None
                self.selected_organization = None
                self.reject()
        else:
            # Login was cancelled via close button - return to role selection
            print("DEBUG: Login dialog was cancelled or rejected - returning to role selection")
            # Don't set role/org to None, keep the role selection dialog open
            # User can choose a different option or close the main dialog


class SignUpDialog(QDialog, BaseDialogMixin):
    """Sign-up dialog that matches the provided image design"""
    
    def __init__(self, parent=None, role="", organization=""):
        super().__init__(parent)
        self.role = role
        self.organization = organization
        self.signup_successful = False
        self.user_data = {}
        self.init_ui()
    
    def init_ui(self):
        """Initialize the sign-up dialog UI"""
        from PyQt5.QtWidgets import QScrollArea

        self.setWindowTitle("Sign Up")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setFixedSize(620, 760)
        self.setModal(True)
        
        # Match the "User Login" visual style
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #161a31, stop:0.55 #1d2444, stop:1 #101427);
            }
            QLabel {
                color: #f5f1ea;
                font-size: 15px;
                font-weight: 600;
                background: transparent;
                border: none;
            }
            QLineEdit {
                padding: 14px 16px;
                border: 1px solid rgba(255,255,255,0.14);
                border-radius: 14px;
                font-size: 14px;
                background: rgba(255,255,255,0.08);
                color: #fffaf5;
                min-height: 22px;
            }
            QLineEdit:focus {
                border: 1px solid #ff8d33;
                background: rgba(255,255,255,0.12);
            }
            QLineEdit::placeholder {
                color: rgba(255,255,255,0.45);
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff6a00, stop:1 #ff9533);
                color: white;
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 14px;
                padding: 12px;
                font-size: 15px;
                font-weight: bold;
                min-height: 18px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff7b14, stop:1 #ffa347);
            }
            QLabel#titleLabel {
                font-size: 30px;
                font-weight: bold;
                color: #fff8f2;
            }
            QLabel#subtitleLabel {
                font-size: 15px;
                font-weight: 600;
                color: rgba(255,255,255,0.72);
            }
            QLabel#fieldLabel {
                font-size: 15px;
                font-weight: 700;
                color: #fff0e6;
            }
            QFrame#signupCard {
                background: rgba(12,16,30,0.62);
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 24px;
            }
            QPushButton#toggleBtn {
                background: #6c757d;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 15px;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton#toggleBtn:hover {
                background: #5a6268;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                border: none;
                background: transparent;
                width: 10px;
                margin: 8px 4px 8px 4px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255,255,255,0.20);
                min-height: 40px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255,255,255,0.28);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)
        
        outer = QVBoxLayout(self)
        outer.setSpacing(0)
        outer.setContentsMargins(24, 24, 24, 24)

        card = QFrame()
        card.setObjectName("signupCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(32, 30, 32, 28)
        card_layout.setSpacing(12)
        outer.addWidget(card)

        title = QLabel("Create Account")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(title)

        subtitle = QLabel(f"{self.role} - {self.organization}")
        subtitle.setObjectName("subtitleLabel")
        subtitle.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        try:
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.viewport().setStyleSheet("background: transparent;")
        except Exception:
            pass

        body = QWidget()
        body.setStyleSheet("background: transparent;")
        form = QVBoxLayout(body)
        form.setContentsMargins(0, 10, 0, 0)
        form.setSpacing(10)
        scroll.setWidget(body)
        card_layout.addWidget(scroll, 1)

        def _field_label(text):
            lbl = QLabel(text)
            lbl.setObjectName("fieldLabel")
            return lbl

        def _text_field(placeholder, max_width=560):
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            edit.setMinimumSize(400, 40)
            edit.setMaximumWidth(max_width)
            edit.setClearButtonEnabled(True)
            return edit

        form.addWidget(_field_label("Full Name:"))
        self.full_name_edit = _text_field("Enter your full name")
        form.addWidget(self.full_name_edit)

        form.addWidget(_field_label("Age:"))
        self.age_edit = _text_field("Enter your age", max_width=320)
        self.age_edit.setValidator(QIntValidator(0, 150, self))
        self.age_edit.setMaxLength(3)
        form.addWidget(self.age_edit)

        form.addWidget(_field_label("Gender:"))
        self.gender_edit = _text_field("Enter your gender", max_width=420)
        form.addWidget(self.gender_edit)

        form.addWidget(_field_label("Address:"))
        self.address_edit = _text_field("Enter your address")
        form.addWidget(self.address_edit)

        form.addWidget(_field_label("Phone Number:"))
        self.phone_edit = _text_field("Enter your phone number", max_width=420)
        # QIntValidator(0, 2147483647) refused every keystroke of any number
        # above 2147483647 — which is most real mobile numbers, since they start
        # at 6. A digit regex has no integer ceiling.
        from utils.input_validation import apply_digit_only, PHONE_DIGITS
        apply_digit_only(self.phone_edit, PHONE_DIGITS)
        form.addWidget(self.phone_edit)

        def _toggle_visibility(pwd_field, btn):
            try:
                if pwd_field.echoMode() == QLineEdit.Password:
                    pwd_field.setEchoMode(QLineEdit.Normal)
                    btn.setText("\U0001F512")  # 🔒
                else:
                    pwd_field.setEchoMode(QLineEdit.Password)
                    btn.setText("\U0001F441")  # 👁
            except Exception:
                pass

        form.addWidget(_field_label("Password:"))
        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("Enter password")
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setMinimumSize(400, 40)
        self.password_edit.setMaximumWidth(520)
        pwd_row = QHBoxLayout()
        pwd_row.setSpacing(12)
        pwd_row.addWidget(self.password_edit, 1)
        self.password_eye_btn = QPushButton("\U0001F441")  # 👁
        self.password_eye_btn.setObjectName("toggleBtn")
        self.password_eye_btn.setFixedSize(40, 40)
        self.password_eye_btn.clicked.connect(lambda checked=False: _toggle_visibility(self.password_edit, self.password_eye_btn))
        pwd_row.addWidget(self.password_eye_btn)
        form.addLayout(pwd_row)

        form.addWidget(_field_label("Confirm Password:"))
        self.confirm_password_edit = QLineEdit()
        self.confirm_password_edit.setPlaceholderText("Confirm password")
        self.confirm_password_edit.setEchoMode(QLineEdit.Password)
        self.confirm_password_edit.setMinimumSize(400, 40)
        self.confirm_password_edit.setMaximumWidth(520)
        self.confirm_password_edit.returnPressed.connect(self.handle_signup)
        confirm_row = QHBoxLayout()
        confirm_row.setSpacing(12)
        confirm_row.addWidget(self.confirm_password_edit, 1)
        self.confirm_eye_btn = QPushButton("\U0001F441")  # 👁
        self.confirm_eye_btn.setObjectName("toggleBtn")
        self.confirm_eye_btn.setFixedSize(40, 40)
        self.confirm_eye_btn.clicked.connect(lambda checked=False: _toggle_visibility(self.confirm_password_edit, self.confirm_eye_btn))
        confirm_row.addWidget(self.confirm_eye_btn)
        form.addLayout(confirm_row)

        card_layout.addSpacing(6)
        self.signup_btn = QPushButton("Sign Up")
        self.signup_btn.clicked.connect(self.handle_signup)
        self.signup_btn.setMinimumHeight(52)
        card_layout.addWidget(self.signup_btn)
    
    def toggle_password_visibility(self, password_field, eye_button):
        """Toggle password visibility between hidden and visible"""
        if password_field.echoMode() == QLineEdit.Password:
            password_field.setEchoMode(QLineEdit.Normal)
            eye_button.setText("\U0001F512")  # 🔒
        else:
            password_field.setEchoMode(QLineEdit.Password)
            eye_button.setText("\U0001F441")  # 👁
    
    def handle_signup(self):
        """Handle the sign-up process"""
        # Get form data
        full_name = self.full_name_edit.text().strip()
        age = self.age_edit.text().strip()
        gender = self.gender_edit.text().strip()
        address = self.address_edit.text().strip()
        phone = self.phone_edit.text().strip()
        password = self.password_edit.text()
        confirm_password = self.confirm_password_edit.text()
        
        # Validate fields
        if not all([full_name, age, gender, address, phone, password, confirm_password]):
            QMessageBox.warning(self, "Error", "All fields are required.")
            return

        # Shared limits — the same rules the login, signup and analysis forms
        # use, so a value accepted in one place cannot be refused in another.
        # "at most 10 digits" previously admitted a 1-digit phone number, and
        # age had no upper bound at all.
        from utils.input_validation import (
            validate_age, validate_name, validate_password, validate_phone,
            validate_text, ADDRESS_MAX_LENGTH,
        )

        checks = (
            validate_name(full_name, "Full name"),
            validate_age(age, "Age"),
            validate_name(gender, "Gender", min_length=1, max_length=20),
            validate_text(address, "Address", ADDRESS_MAX_LENGTH),
            validate_phone(phone),
        )
        for ok, _cleaned, err in checks:
            if not ok:
                QMessageBox.warning(self, "Error", err)
                return

        full_name = checks[0][1]
        age = checks[1][1]
        gender = checks[2][1]
        address = checks[3][1]
        phone = checks[4][1]

        if password != confirm_password:
            QMessageBox.warning(self, "Error", "Passwords do not match.")
            return

        ok_pwd, password, err_pwd = validate_password(password)
        if not ok_pwd:
            QMessageBox.warning(self, "Error", err_pwd)
            return

        # Store user data
        self.user_data = {
            'full_name': full_name,
            'age': age,
            'gender': gender,
            'address': address,
            'phone': phone,
            'password': password,
            'role': self.role,
            'organization': self.organization,
            'signup_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to appropriate file based on role
        try:
            print("DEBUG: Attempting to save head user data...")
            
            if self.role == 'Doctor Head':
                doctor_head_users = load_doctor_head_users()
                username = phone  # Use phone as unique identifier
                
                if username in doctor_head_users:
                    print(f"DEBUG: Doctor Head user {username} already exists")
                    QMessageBox.warning(self, "Error", "A user with this phone number already exists.")
                    return
                
                doctor_head_users[username] = self.user_data
                save_head_users(doctor_head_users, 'Doctor Head')
                
                # Also save to main users.json for normal login
                save_user_to_main_users(self.user_data, username)
            elif self.role == 'HCP Head':
                hcp_head_users = load_hcp_head_users()
                username = phone  # Use phone as unique identifier
                
                if username in hcp_head_users:
                    print(f"DEBUG: HCP Head user {username} already exists")
                    QMessageBox.warning(self, "Error", "A user with this phone number already exists.")
                    return
                
                hcp_head_users[username] = self.user_data
                save_head_users(hcp_head_users, 'HCP Head')
                
                # Also save to main users.json for normal login
                save_user_to_main_users(self.user_data, username)
            
            self.signup_successful = True
            print("DEBUG: signup_successful set to True")
            QMessageBox.information(self, "Success", "Sign-up successful! Your dashboard will now open.")
            self.accept()
            
        except Exception as e:
            print(f"DEBUG: Exception during head user signup: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save user data: {str(e)}")
    
    def get_user_data(self):
        """Return the signed-up user data"""
        result = self.user_data if self.signup_successful else None
        print(f"DEBUG: get_user_data called. signup_successful={self.signup_successful}, result={result}")
        return result


class LoginDialog(QDialog, BaseDialogMixin):
    """Login dialog for existing Head Doctor and Head HCP users"""
    
    def __init__(self, parent=None, role="", organization=""):
        super().__init__(parent)
        self.role = role
        self.organization = organization
        self.login_successful = False
        self.user_data = {}
        self.init_ui()
    
    def init_ui(self):
        """Initialize the login dialog UI"""
        self.setWindowTitle("Login")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setFixedSize(620, 560)
        self.setModal(True)
        
        # Match the "User Login" visual style, but keep a black secondary button
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #161a31, stop:0.55 #1d2444, stop:1 #101427);
            }
            QLabel {
                color: #f5f1ea;
                font-size: 15px;
                font-weight: 600;
                background: transparent;
                border: none;
            }
            QLabel#titleLabel {
                font-size: 30px;
                font-weight: bold;
                color: #fff8f2;
            }
            QLabel#subtitleLabel {
                font-size: 15px;
                font-weight: 600;
                color: rgba(255,255,255,0.72);
            }
            QLineEdit {
                padding: 14px 16px;
                border: 1px solid rgba(255,255,255,0.14);
                border-radius: 14px;
                font-size: 14px;
                background: rgba(255,255,255,0.08);
                color: #fffaf5;
                min-height: 22px;
            }
            QLineEdit:focus {
                border: 1px solid #ff8d33;
                background: rgba(255,255,255,0.12);
            }
            QLineEdit::placeholder {
                color: rgba(255,255,255,0.45);
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff6a00, stop:1 #ff9533);
                color: white;
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 14px;
                padding: 12px;
                font-size: 15px;
                font-weight: bold;
                min-height: 18px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff7b14, stop:1 #ffa347);
            }
            QPushButton#toggleBtn {
                background: #6c757d;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 15px;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton#toggleBtn:hover {
                background: #5a6268;
            }
            QPushButton#phoneLoginBtn {
                background: rgba(0,0,0,0.40);
                color: #ffffff;
                border: 1px solid rgba(255,255,255,0.12);
            }
            QPushButton#phoneLoginBtn:hover {
                background: rgba(0,0,0,0.55);
            }
            QFrame#headLoginCard {
                background: rgba(12,16,30,0.62);
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 24px;
            }
        """)

        outer = QVBoxLayout(self)
        outer.setSpacing(0)
        outer.setContentsMargins(24, 24, 24, 24)

        card = QFrame()
        card.setObjectName("headLoginCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(32, 30, 32, 28)
        card_layout.setSpacing(12)
        outer.addWidget(card)

        title = QLabel("Login to Account")
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("titleLabel")
        card_layout.addWidget(title)

        subtitle = QLabel(f"{self.role} - {self.organization}")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setObjectName("subtitleLabel")
        card_layout.addWidget(subtitle)

        full_name_label = QLabel("Full Name:")
        full_name_label.setObjectName("fieldLabel")
        card_layout.addWidget(full_name_label)

        self.full_name_edit = QLineEdit()
        self.full_name_edit.setPlaceholderText("Enter your full name")
        self.full_name_edit.returnPressed.connect(self.handle_login)
        card_layout.addWidget(self.full_name_edit)

        password_label = QLabel("Password:")
        password_label.setObjectName("fieldLabel")
        card_layout.addWidget(password_label)

        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("Enter your password")
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.returnPressed.connect(self.handle_login)

        password_row = QHBoxLayout()
        password_row.setSpacing(12)
        password_row.addWidget(self.password_edit, 1)
        self.password_eye_btn = QPushButton("\U0001F441")  # 👁
        self.password_eye_btn.setObjectName("toggleBtn")
        self.password_eye_btn.setFixedSize(40, 40)
        self.password_eye_btn.clicked.connect(lambda checked=False: self.toggle_password_visibility(self.password_edit, self.password_eye_btn))
        password_row.addWidget(self.password_eye_btn)
        card_layout.addLayout(password_row)

        card_layout.addSpacing(8)
        self.login_btn = QPushButton("Login")
        self.login_btn.clicked.connect(self.handle_login)
        self.login_btn.setMinimumHeight(52)
        card_layout.addWidget(self.login_btn)

        self.phone_login_btn = QPushButton("Login with Phone Number")
        self.phone_login_btn.setObjectName("phoneLoginBtn")
        self.phone_login_btn.clicked.connect(self.handle_phone_login)
        self.phone_login_btn.setMinimumHeight(52)
        card_layout.addWidget(self.phone_login_btn)

        card_layout.addStretch()
    
    def toggle_password_visibility(self, password_field, eye_button):
        """Toggle password visibility between hidden and visible"""
        if password_field.echoMode() == QLineEdit.Password:
            password_field.setEchoMode(QLineEdit.Normal)
            eye_button.setText("🔒")
        else:
            password_field.setEchoMode(QLineEdit.Password)
            eye_button.setText("👁")
    
        try:
            if password_field.echoMode() == QLineEdit.Normal:
                eye_button.setText("\U0001F512")  # 🔒
            else:
                eye_button.setText("\U0001F441")  # 👁
        except Exception:
            pass

        # Final correction for mojibake icons
        if password_field.echoMode() == QLineEdit.Normal:
            eye_button.setText("\U0001F512")  # 🔒
        else:
            eye_button.setText("\U0001F441")  # 👁

    def handle_login(self):
        """Handle the login process"""
        full_name = self.full_name_edit.text().strip()
        password = self.password_edit.text()
        
        if not full_name or not password:
            QMessageBox.warning(self, "Error", "Both full name and password are required.")
            return
        
        # Check credentials against appropriate file for Head roles
        try:
            print(f"DEBUG: Looking for full_name='{full_name}', role='{self.role}', org='{self.organization}'")
            
            if self.role == 'Doctor Head':
                head_users = load_doctor_head_users()
            elif self.role == 'HCP Head':
                head_users = load_hcp_head_users()
            else:
                head_users = {}
                
            print(f"DEBUG: Available head users: {head_users}")
            
            # Find user by full name and role
            found_user = None
            for username, user_record in head_users.items():
                record_full_name = user_record.get('full_name', '').lower()
                record_role = user_record.get('role', '')
                record_org = user_record.get('organization', '')
                input_full_name = full_name.lower()
                
                print(f"DEBUG: Checking head user {username}: full_name='{record_full_name}' vs '{input_full_name}', role='{record_role}' vs '{self.role}', org='{record_org}' vs '{self.organization}'")
                
                if (record_full_name == input_full_name and 
                    record_role == self.role and
                    record_org == self.organization):
                    found_user = user_record
                    print(f"DEBUG: Found matching head user: {found_user}")
                    break
            
            if found_user and found_user.get('password') == password:
                print(f"DEBUG: Password matches for head user")
                self.user_data = found_user
                self.login_successful = True
                QMessageBox.information(self, "Success", "Login successful! Your dashboard will now open.")
                self.accept()
            else:
                if not found_user:
                    print(f"DEBUG: No head user found with matching credentials")
                else:
                    print(f"DEBUG: Password mismatch. Expected: {found_user.get('password')}, Got: {password}")
                QMessageBox.warning(self, "Error", "Invalid credentials. Please check your full name and password.")
                
        except Exception as e:
            print(f"DEBUG: Exception during head user login: {e}")
            QMessageBox.critical(self, "Error", f"Login failed: {str(e)}")
    
    def handle_phone_login(self):
        """Handle phone number login"""
        # Create phone input dialog
        phone_dialog = QDialog(self)
        phone_dialog.setWindowTitle("Login with Phone Number")
        phone_dialog.setWindowFlags(phone_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        phone_dialog.setFixedSize(620, 360)
        phone_dialog.setModal(True)
        phone_dialog.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #161a31, stop:0.55 #1d2444, stop:1 #101427);
            }
            QLabel {
                color: #f5f1ea;
                font-size: 14px;
                font-weight: 600;
                background: transparent;
                border: none;
            }
            QLabel#titleLabel {
                font-size: 30px;
                font-weight: bold;
                color: #fff8f2;
            }
            QLabel#subtitleLabel {
                font-size: 15px;
                font-weight: 600;
                color: rgba(255,255,255,0.72);
            }
            QLineEdit {
                padding: 14px 16px;
                border: 1px solid rgba(255,255,255,0.14);
                border-radius: 14px;
                font-size: 14px;
                background: rgba(255,255,255,0.08);
                color: #fffaf5;
                min-height: 22px;
            }
            QLineEdit:focus {
                border: 1px solid #ff8d33;
                background: rgba(255,255,255,0.12);
            }
            QLineEdit::placeholder {
                color: rgba(255,255,255,0.45);
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff6a00, stop:1 #ff9533);
                color: white;
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 14px;
                padding: 12px;
                font-size: 15px;
                font-weight: bold;
                min-height: 18px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff7b14, stop:1 #ffa347);
            }
            QPushButton#cancelBtn {
                background: rgba(0,0,0,0.40);
                color: #fff7ef;
            }
            QPushButton#cancelBtn:hover {
                background: rgba(0,0,0,0.55);
            }
            QFrame#phoneLoginCard {
                background: rgba(12,16,30,0.62);
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 24px;
            }
        """)

        layout = QVBoxLayout(phone_dialog)
        layout.setSpacing(0)
        layout.setContentsMargins(24, 24, 24, 24)

        card = QFrame()
        card.setObjectName("phoneLoginCard")
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(18)
        card_layout.setContentsMargins(32, 30, 32, 28)
        layout.addWidget(card)

        title = QLabel("Login to Continue")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(title)

        subtitle = QLabel(f"{self.role} - {self.organization}")
        subtitle.setObjectName("subtitleLabel")
        subtitle.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(subtitle)

        label = QLabel("Phone Number:")
        card_layout.addWidget(label)

        phone_edit = QLineEdit()
        phone_edit.setPlaceholderText("Enter your phone number")
        from utils.input_validation import apply_digit_only, PHONE_DIGITS
        apply_digit_only(phone_edit, PHONE_DIGITS)
        phone_edit.returnPressed.connect(lambda: do_phone_login())
        card_layout.addWidget(phone_edit)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(14)
        login_btn = QPushButton("Login")
        login_btn.setMinimumHeight(52)
        login_btn.setAutoDefault(True)
        login_btn.setDefault(True)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("cancelBtn")
        cancel_btn.setMinimumHeight(52)
        
        def do_phone_login():
            phone = phone_edit.text().strip()
            if not phone:
                QMessageBox.warning(phone_dialog, "Error", "Phone number is required.")
                return
            
            # Check credentials against appropriate file for phone login
            try:
                if self.role == 'Doctor Head':
                    head_users = load_doctor_head_users()
                elif self.role == 'HCP Head':
                    head_users = load_hcp_head_users()
                else:
                    head_users = {}
                    
                print(f"DEBUG: Available head users for phone login: {head_users}")
                
                # Find user by phone and role
                found_user = None
                for username, user_record in head_users.items():
                    if (user_record.get('phone', '') == phone and 
                        user_record.get('role', '') == self.role and
                        user_record.get('organization', '') == self.organization):
                        found_user = user_record
                        break
                
                if found_user:
                    self.user_data = found_user
                    self.login_successful = True
                    phone_dialog.accept()
                    self.accept()
                else:
                    QMessageBox.warning(phone_dialog, "Error", "No account found with this phone number.")
                    
            except Exception as e:
                QMessageBox.critical(phone_dialog, "Error", f"Login failed: {str(e)}")
        
        login_btn.clicked.connect(do_phone_login)
        cancel_btn.clicked.connect(phone_dialog.reject)
        
        button_layout.addWidget(login_btn)
        button_layout.addWidget(cancel_btn)
        card_layout.addSpacing(6)
        card_layout.addLayout(button_layout)
        
        if phone_dialog.exec_() == QDialog.Accepted:
            QMessageBox.information(self, "Success", "Login successful! Your dashboard will now open.")
    
    def get_user_data(self):
        """Return the logged-in user data"""
        return self.user_data if self.login_successful else None


class DashboardWindow(QDialog):
    """Full-screen dashboard that opens after successful sign-up"""
    
    def __init__(self, parent=None, user_data=None):
        super().__init__(parent)
        self.user_data = user_data or {}
        self.init_ui()
    
    def keyPressEvent(self, event):
        """Handle key press events to prevent Enter key from closing dialog"""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # Check if the focus widget is a QLineEdit that has returnPressed connected
            focus_widget = self.focusWidget()
            if isinstance(focus_widget, QLineEdit):
                # Let the QLineEdit handle the Enter key (for returnPressed signals)
                focus_widget.keyPressEvent(event)
                return
            # Otherwise, ignore Enter key to prevent accidental dialog closure
            event.ignore()
            return
        # Handle other keys normally
        super().keyPressEvent(event)

    def init_ui(self):
        """Initialize the dashboard UI"""
        self.setWindowTitle("Dashboard")
        self.setModal(True)  # Modal to prevent background interaction
        
        # Enable window flags for better window management
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)

        # Full screen
        self.showMaximized()
        
        # Disable default button behavior to prevent Enter key from closing dialog
        self.buttonGroup = None
        
        # Set background gradient
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #1a1a2e, stop:1 #16213e);
                color: white;
            }
            QLabel {
                color: white;
                font-size: 14px;
            }
            QPushButton {
                background: #ff6600;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #ff8800;
            }
            QFrame {
                background: rgba(255,255,255,0.1);
                border-radius: 12px;
                border: 1px solid rgba(255,255,255,0.2);
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        header_layout = QHBoxLayout()
        
        # Welcome message
        welcome_label = QLabel(f"Welcome, {self.user_data.get('full_name', 'User')}!")
        welcome_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #ff6600;")
        header_layout.addWidget(welcome_label)
        
        header_layout.addStretch()
        
        # Role and organization info
        role_org_label = QLabel(f"{self.user_data.get('role', 'N/A')} - {self.user_data.get('organization', 'N/A')}")
        role_org_label.setStyleSheet("font-size: 16px; color: #ccc;")
        header_layout.addWidget(role_org_label)
        
        # Logout button
        logout_btn = QPushButton("Logout")
        logout_btn.clicked.connect(self.close)
        logout_btn.setAutoDefault(False)  # Prevent this button from being triggered by Enter
        logout_btn.setDefault(False)  # Explicitly set as non-default
        header_layout.addWidget(logout_btn)
        
        layout.addLayout(header_layout)
        
        # Main content area
        main_content = QHBoxLayout()
        
        # Left sidebar - User info card
        left_frame = QFrame()
        left_frame.setFixedWidth(300)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(20, 20, 20, 20)
        
        # User info title
        user_info_title = QLabel("User Information")
        user_info_title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 15px;")
        left_layout.addWidget(user_info_title)
        
        # User details
        user_details = [
            ("Full Name:", self.user_data.get('full_name', 'N/A')),
            ("Age:", self.user_data.get('age', 'N/A')),
            ("Gender:", self.user_data.get('gender', 'N/A')),
            ("Phone:", self.user_data.get('phone', 'N/A')),
            ("Address:", self.user_data.get('address', 'N/A')),
            ("Role:", self.user_data.get('role', 'N/A')),
            ("Organization:", self.user_data.get('organization', 'N/A')),
            ("Member Since:", self.user_data.get('signup_date', 'N/A'))
        ]
        
        for label, value in user_details:
            detail_layout = QHBoxLayout()
            label_widget = QLabel(label)
            label_widget.setStyleSheet("font-weight: bold; color: #ff6600;")
            value_widget = QLabel(value)
            value_widget.setWordWrap(True)
            detail_layout.addWidget(label_widget)
            detail_layout.addWidget(value_widget)
            detail_layout.addStretch()
        
        # Add Users button for Doctor Head and HCP Head
        if self.user_data.get('role') in ['Doctor Head', 'HCP Head']:
            add_users_btn = QPushButton("Add Users")
            add_users_btn.setStyleSheet("""
                QPushButton {
                    background: #28a745;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 12px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: #218838;
                }
            """)
            add_users_btn.clicked.connect(self.show_add_users_dialog)
            add_users_btn.setAutoDefault(False)  # Prevent this button from being triggered by Enter
            add_users_btn.setDefault(False)  # Explicitly set as non-default
            left_layout.addWidget(add_users_btn)
            left_layout.addSpacing(10)
            
            # User count display for Doctor Head and HCP Head
            if self.user_data.get('role') in ['Doctor Head', 'HCP Head']:
                user_count = self.get_user_count()
                count_label = QLabel(f"Users: {user_count}")
                count_label.setStyleSheet("""
                    QLabel {
                        background: rgba(40, 167, 69, 0.2);
                        color: #28a745;
                        border-radius: 6px;
                        padding: 8px 12px;
                        font-size: 12px;
                        font-weight: bold;
                    }
                """)
                count_label.setAlignment(Qt.AlignCenter)
                self.user_count_label = count_label
                left_layout.addWidget(count_label)
        
        left_layout.addStretch()
        
        main_content.addWidget(left_frame)
        
        # Center area - Main dashboard content
        center_frame = QFrame()
        center_layout = QVBoxLayout(center_frame)
        center_layout.setContentsMargins(30, 30, 30, 30)
        
        # Dashboard title
        dashboard_title = QLabel("Dashboard Overview")
        dashboard_title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        center_layout.addWidget(dashboard_title)
        
        # Quick stats
        stats_layout = QHBoxLayout()
        
        # Sample stats cards
        stats_data = [
            ("Total Patients", "0", "#007bff"),
            ("Appointments Today", "0", "#28a745"),
            ("Pending Reports", "0", "#ffc107"),
            ("Messages", "0", "#dc3545")
        ]
        
        for title, value, color in stats_data:
            stat_card = QFrame()
            stat_card.setStyleSheet(f"""
                QFrame {{
                    background: rgba(255,255,255,0.05);
                    border-radius: 8px;
                    border: none;
                }}
                QLabel {{
                    color: white;
                }}
            """)
            stat_card.setMinimumWidth(200)
            stat_layout_inner = QVBoxLayout(stat_card)
            stat_layout_inner.setContentsMargins(15, 15, 15, 15)
            
            stat_value = QLabel(value)
            stat_value.setStyleSheet(f"font-size: 32px; font-weight: bold; color: {color};")
            stat_value.setAlignment(Qt.AlignCenter)
            stat_layout_inner.addWidget(stat_value)
            
            stat_title = QLabel(title)
            stat_title.setStyleSheet("font-size: 14px; margin-top: 5px;")
            stat_title.setAlignment(Qt.AlignCenter)
            stat_layout_inner.addWidget(stat_title)
            
            stats_layout.addWidget(stat_card)
        
        center_layout.addLayout(stats_layout)
        center_layout.addSpacing(30)
        
        # Recent activity section
        activity_title = QLabel("Recent Activity")
        activity_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #fff7ef; margin-bottom: 8px;")
        center_layout.addWidget(activity_title)
        
        # Load and display recently created users
        recent_users_frame = QFrame()
        recent_users_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255,255,255,0.06),
                    stop:1 rgba(255,255,255,0.03));
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 22px;
            }
        """)
        recent_users_layout = QVBoxLayout(recent_users_frame)
        recent_users_layout.setContentsMargins(18, 18, 18, 18)
        recent_users_layout.setSpacing(12)
        self.recent_users_layout = recent_users_layout
        self._populate_recent_users()
        
        center_layout.addWidget(recent_users_frame)
        
        center_layout.addStretch()
        main_content.addWidget(center_frame, 1)
        
        layout.addLayout(main_content)
        
        # Footer
        footer_label = QLabel("© 2026 CardioX by Deckmount. All rights reserved.")
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setStyleSheet("color: #666; font-size: 12px; margin-top: 20px;")
        layout.addWidget(footer_label)
    
    def show_add_users_dialog(self):
        """Show dialog for adding users based on role hierarchy"""
        dialog = AddUsersDialog(self, self.user_data)
        dialog.exec_()
        
        # Refresh recent activity after adding users
        self.refresh_recent_activity()
    
    def get_recent_users(self):
        """Get recently created users by current Doctor Head/HCP Head in current organization"""
        try:
            current_role = self.user_data.get('role', '')
            creator_name = self.user_data.get('full_name', '')
            current_organization = self.user_data.get('organization', '')
            
            if current_role in ['Doctor Head', 'HCP Head']:
                # Get users from creator-specific file
                created_users = load_created_users(creator_name, current_role)
                
                # Convert to list and filter by current organization
                users_list = []
                for username, user_data in created_users.items():
                    # Only include users created in the current organization
                    if user_data.get('organization') == current_organization:
                        user_data['username'] = username
                        users_list.append(user_data)
                    else:
                        print(f"DEBUG: Skipping user {username} - org {user_data.get('organization')} != current org {current_organization}")
                
                # Sort by signup_date
                users_list.sort(key=lambda x: x.get('signup_date', ''), reverse=True)
                
                # Return only last 5 users
                return users_list[:5]
            else:
                return []
        except Exception as e:
            print(f"DEBUG: Error getting recent users: {e}")
            return []
    
    def create_user_card(self, user_data):
        """Create a clickable user card"""
        from PyQt5.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy, QGraphicsDropShadowEffect
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QColor
        
        card = QFrame()
        card.setMinimumSize(350, 80)
        card.setMaximumHeight(100)
        card.setStyleSheet("""
            QFrame {
                background: rgba(30,30,40,0.8);
                border: 2px solid white;
                border-radius: 10px;
                padding: 12px;
            }
            QFrame:hover {
                background: rgba(40,40,50,0.9);
                border: 2px solid red;
            }
        """)
        
        layout = QHBoxLayout(card)
        layout.setContentsMargins(15, 10, 15, 10)
        
        # User info
        info_layout = QVBoxLayout()
        
        # Name and role
        name_label = QLabel(f"{user_data.get('full_name', 'N/A')}")
        name_label.setStyleSheet("font-size: 15px; font-weight: bold; color: white; margin-bottom: 3px;")
        name_label.setWordWrap(True)
        name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        info_layout.addWidget(name_label)
        
        role_label = QLabel(f"{user_data.get('role', 'N/A')} • {user_data.get('phone', 'N/A')}")
        role_label.setStyleSheet("font-size: 13px; color: #bbb;")
        role_label.setWordWrap(True)
        role_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        info_layout.addWidget(role_label)
        
        layout.addLayout(info_layout)
        layout.addStretch()
        
        # Click handler for login
        def show_login_dialog():
            login_dialog = UserLoginDialog(self, user_data)
            if login_dialog.exec_() == QDialog.Accepted:
                # Login successful - close current sign-up page and start main application
                self.start_main_application(login_dialog.user_data)
        
        # Make card clickable
        from PyQt5.QtWidgets import QPushButton
        card_btn = QPushButton()
        card_btn.setStyleSheet("QPushButton { border: none; background: transparent; }")
        card_btn.setCursor(Qt.PointingHandCursor)
        card_btn.clicked.connect(show_login_dialog)
        card_btn.setLayout(layout)
        
        return card_btn

    def create_recent_activity_card(self, user_data):
        """Create a polished recent activity card with the same user details"""
        from PyQt5.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy, QGraphicsDropShadowEffect, QPushButton
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QColor

        full_name = user_data.get('full_name', 'N/A')
        role_text = user_data.get('role', 'N/A')
        phone_text = user_data.get('phone', 'N/A')

        initials_parts = [part[0].upper() for part in full_name.split() if part]
        initials = "".join(initials_parts[:2]) or "NA"

        accent_options = [
            ("#6a338f", "#8d5bc6", "#f3e7ff"),
            ("#2e98ab", "#92d4dc", "#ebfdff"),
            ("#ff7849", "#ffb06a", "#fff1e8"),
            ("#2f7b65", "#67bea4", "#e8fff7"),
        ]
        accent_seed = sum(ord(ch) for ch in f"{full_name}{role_text}")
        accent_color, accent_soft, accent_bg = accent_options[accent_seed % len(accent_options)]

        card = QFrame()
        card.setCursor(Qt.PointingHandCursor)
        card.setMinimumHeight(122)
        card.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {accent_color},
                    stop:1 {accent_soft});
                border: 1px solid rgba(255,255,255,0.16);
                border-radius: 20px;
            }}
            QFrame:hover {{
                border: 1px solid rgba(255,255,255,0.32);
            }}
            QLabel {{
                background: transparent;
                border: none;
                color: white;
            }}
        """)

        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(26)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(0, 0, 0, 75))
        card.setGraphicsEffect(shadow)

        layout = QHBoxLayout(card)
        layout.setContentsMargins(18, 16, 80, 16)
        layout.setSpacing(14)

        avatar = QLabel(initials)
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setFixedSize(52, 52)
        avatar.setStyleSheet(f"""
            QLabel {{
                background: {accent_bg};
                color: {accent_color};
                border-radius: 26px;
                font-size: 16px;
                font-weight: bold;
            }}
        """)
        layout.addWidget(avatar, 0, Qt.AlignTop)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(6)

        name_label = QLabel(full_name)
        name_label.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        name_label.setWordWrap(True)
        name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        info_layout.addWidget(name_label)

        role_label = QLabel(role_text)
        role_label.setStyleSheet("font-size: 13px; font-weight: 600; color: rgba(255,255,255,0.92);")
        role_label.setWordWrap(True)
        role_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        info_layout.addWidget(role_label)

        phone_label = QLabel(phone_text)
        phone_label.setStyleSheet("""
            font-size: 12px;
            color: rgba(255,255,255,0.84);
            background: rgba(0,0,0,0.14);
            border-radius: 10px;
            padding: 5px 10px;
        """)
        phone_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        info_layout.addWidget(phone_label, 0, Qt.AlignLeft)

        progress_track = QFrame()
        progress_track.setFixedHeight(8)
        progress_track.setStyleSheet("background: rgba(255,255,255,0.24); border-radius: 4px;")
        progress_fill = QFrame(progress_track)
        progress_fill.setStyleSheet("background: rgba(255,255,255,0.96); border-radius: 4px;")

        def resize_progress(event=None):
            fill_width = max(96, int(progress_track.width() * 0.72))
            progress_fill.setGeometry(0, 0, fill_width, progress_track.height())
            if event is not None:
                QFrame.resizeEvent(progress_track, event)

        progress_track.resizeEvent = resize_progress
        info_layout.addWidget(progress_track)
        info_layout.addStretch()

        layout.addLayout(info_layout, 1)

        action_layout = QHBoxLayout()
        action_layout.setSpacing(8)
        action_layout.setContentsMargins(0, 0, 0, 0)

        pill_style = """
            QPushButton {
                color: white;
                font-size: 11px;
                font-weight: bold;
                background: rgba(0,0,0,0.14);
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 14px;
                padding: 0px 10px;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                background: rgba(0,0,0,0.22);
            }
        """

        open_btn = QPushButton("OPEN")
        open_btn.setFixedSize(56, 28)
        open_btn.setCursor(Qt.PointingHandCursor)
        open_btn.setStyleSheet(pill_style)
        action_layout.addWidget(open_btn, 0, Qt.AlignTop)

        view_btn = QPushButton("VIEW")
        view_btn.setFixedSize(56, 28)
        view_btn.setCursor(Qt.PointingHandCursor)
        view_btn.setStyleSheet(pill_style)
        action_layout.addWidget(view_btn, 0, Qt.AlignTop)

        edit_btn = QPushButton("EDIT")
        edit_btn.setFixedSize(56, 28)
        edit_btn.setCursor(Qt.PointingHandCursor)
        edit_btn.setStyleSheet(pill_style)
        action_layout.addWidget(edit_btn, 0, Qt.AlignTop)

        delete_btn = QPushButton("DELETE")
        delete_btn.setFixedSize(96, 30)
        delete_btn.setCursor(Qt.PointingHandCursor)
        delete_btn.setStyleSheet("""
            QPushButton {
                color: white;
                font-size: 11px;
                font-weight: bold;
                background: rgba(190, 40, 40, 0.85);
                border: 1px solid rgba(255,255,255,0.20);
                border-radius: 14px;
                padding: 0px 10px;
            }
            QPushButton:hover {
                background: rgba(220, 60, 60, 0.95);
            }
        """)
        action_layout.addWidget(delete_btn, 0, Qt.AlignTop)

        layout.addLayout(action_layout, 0)
        resize_progress()

        def show_login_dialog():
            login_dialog = UserLoginDialog(self, user_data)
            if login_dialog.exec_() == QDialog.Accepted:
                self.start_main_application(login_dialog.user_data)

        def delete_user_card():
            self.delete_created_user(user_data)

        def view_user_card():
            self.view_created_user_details(user_data)

        def edit_user_card():
            if self.edit_created_user_details(user_data):
                self.refresh_recent_activity()

        open_btn.clicked.connect(show_login_dialog)
        view_btn.clicked.connect(view_user_card)
        edit_btn.clicked.connect(edit_user_card)
        delete_btn.clicked.connect(delete_user_card)

        original_mouse_press = card.mousePressEvent

        def handle_card_press(event):
            if event.button() == Qt.LeftButton:
                show_login_dialog()
                return
            original_mouse_press(event)

        card.mousePressEvent = handle_card_press
        return card

    def view_created_user_details(self, user_data):
        """Show all details for a created sub-user."""
        try:
            from PyQt5.QtWidgets import (
                QDialog,
                QVBoxLayout,
                QHBoxLayout,
                QLabel,
                QPushButton,
                QFrame,
                QScrollArea,
                QWidget,
                QGridLayout,
            )
            from PyQt5.QtCore import Qt

            full_name = str(user_data.get("full_name", "") or "User")
            role_text = str(user_data.get("role", "") or "")
            org_text = str(user_data.get("organization", "") or "")

            details_order = [
                ("Full Name", user_data.get("full_name", "")),
                ("Role", user_data.get("role", "")),
                ("Organization", user_data.get("organization", "")),
                ("Phone", user_data.get("phone", "")),
                ("Email", user_data.get("email", "")),
                ("Age", user_data.get("age", "")),
                ("Gender", user_data.get("gender", "")),
                ("Address", user_data.get("address", "")),
                ("Created By", user_data.get("created_by", "")),
                ("Signup Date", user_data.get("signup_date", "")),
            ]

            initials_parts = [part[0].upper() for part in full_name.split() if part]
            initials = "".join(initials_parts[:2]) or "U"

            dialog = QDialog(self)
            dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            dialog.setWindowTitle("User Details")
            dialog.setModal(True)
            dialog.setMinimumSize(520, 480)
            dialog.setStyleSheet("""
                QDialog {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #131730,
                        stop:0.55 #1b2140,
                        stop:1 #0f1327);
                }
                QLabel {
                    background: transparent;
                    border: none;
                    color: #f7f2ee;
                    font-size: 13px;
                    font-weight: 600;
                }
                QFrame#detailsCard {
                    background: rgba(12,16,30,0.62);
                    border: 1px solid rgba(255,255,255,0.10);
                    border-radius: 22px;
                }
                QLabel#title {
                    font-size: 22px;
                    font-weight: 800;
                    color: #fff8f2;
                }
                QLabel#subtitle {
                    font-size: 12px;
                    font-weight: 600;
                    color: rgba(255,255,255,0.68);
                }
                QLabel#avatar {
                    background: rgba(255, 122, 26, 0.18);
                    border: 1px solid rgba(255, 160, 95, 0.45);
                    color: #ff9a3d;
                    border-radius: 22px;
                    font-size: 16px;
                    font-weight: 800;
                }
                QLabel#key {
                    color: rgba(255,255,255,0.75);
                    font-weight: 700;
                }
                QLabel#value {
                    color: #fff8f2;
                    font-weight: 700;
                }
                QScrollArea {
                    background: transparent;
                    border: none;
                }
                QScrollBar:vertical {
                    background: rgba(255,255,255,0.08);
                    width: 12px;
                    margin: 4px 0 4px 0;
                    border-radius: 6px;
                }
                QScrollBar::handle:vertical {
                    background: rgba(255, 122, 26, 0.9);
                    min-height: 34px;
                    border-radius: 6px;
                }
                QScrollBar::handle:vertical:hover {
                    background: rgba(255, 145, 54, 1.0);
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                }
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                    background: transparent;
                }
                QPushButton#okBtn {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #ff6a00, stop:1 #ff9533);
                    color: white;
                    border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 14px;
                    padding: 10px 18px;
                    font-size: 14px;
                    font-weight: 800;
                    min-width: 110px;
                    min-height: 42px;
                }
                QPushButton#okBtn:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #ff7b14, stop:1 #ffa347);
                }
            """)

            root = QVBoxLayout(dialog)
            root.setContentsMargins(22, 22, 22, 22)
            root.setSpacing(14)

            card = QFrame()
            card.setObjectName("detailsCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(22, 20, 22, 18)
            card_layout.setSpacing(14)
            root.addWidget(card)

            header_row = QHBoxLayout()
            header_row.setSpacing(12)

            avatar = QLabel(initials)
            avatar.setObjectName("avatar")
            avatar.setAlignment(Qt.AlignCenter)
            avatar.setFixedSize(44, 44)
            header_row.addWidget(avatar, 0, Qt.AlignTop)

            header_text = QVBoxLayout()
            header_text.setSpacing(2)

            title = QLabel("User Details")
            title.setObjectName("title")
            header_text.addWidget(title)

            subtitle_parts = [part for part in [role_text, org_text] if part]
            subtitle = QLabel(" • ".join(subtitle_parts) if subtitle_parts else "Created account")
            subtitle.setObjectName("subtitle")
            header_text.addWidget(subtitle)

            header_row.addLayout(header_text, 1)
            card_layout.addLayout(header_row)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            try:
                scroll.viewport().setStyleSheet("background: transparent;")
            except Exception:
                pass

            scroll_body = QWidget()
            scroll_body.setStyleSheet("background: transparent;")
            grid = QGridLayout(scroll_body)
            grid.setContentsMargins(6, 6, 6, 6)
            grid.setHorizontalSpacing(14)
            grid.setVerticalSpacing(10)

            row_idx = 0
            for key, value in details_order:
                safe_value = "" if value is None else str(value)
                key_label = QLabel(f"{key}:")
                key_label.setObjectName("key")
                value_label = QLabel(safe_value if safe_value else "—")
                value_label.setObjectName("value")
                value_label.setWordWrap(True)

                grid.addWidget(key_label, row_idx, 0, Qt.AlignTop)
                grid.addWidget(value_label, row_idx, 1)
                row_idx += 1

            grid.setColumnStretch(0, 0)
            grid.setColumnStretch(1, 1)
            scroll.setWidget(scroll_body)
            card_layout.addWidget(scroll, 1)

            footer_row = QHBoxLayout()
            footer_row.addStretch()
            ok_btn = QPushButton("OK")
            ok_btn.setObjectName("okBtn")
            ok_btn.clicked.connect(dialog.accept)
            footer_row.addWidget(ok_btn)
            card_layout.addLayout(footer_row)

            dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to show user details: {e}")

    def edit_created_user_details(self, user_data):
        """Edit and update an existing created sub-user across storage."""
        try:
            from PyQt5.QtWidgets import (
                QDialog,
                QVBoxLayout,
                QHBoxLayout,
                QGridLayout,
                QLabel,
                QLineEdit,
                QPushButton,
                QFrame,
                QScrollArea,
                QWidget,
            )
            from PyQt5.QtGui import QIntValidator

            original_username = user_data.get("username") or user_data.get("phone") or ""
            if not original_username:
                QMessageBox.warning(self, "Error", "Could not identify this user record for editing.")
                return False

            dialog = QDialog(self)
            dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            dialog.setWindowTitle("Edit User")
            dialog.setMinimumSize(760, 780)
            dialog.setModal(True)
            dialog.setStyleSheet("""
                QDialog {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #161a31, stop:0.55 #1d2444, stop:1 #101427);
                }
                QLabel {
                    color: #f5f1ea;
                    font-size: 15px;
                    font-weight: 600;
                    background: transparent;
                    border: none;
                }
                QLabel#titleLabel {
                    font-size: 30px;
                    font-weight: bold;
                    color: #fff8f2;
                }
                QLabel#subtitleLabel {
                    font-size: 15px;
                    font-weight: 600;
                    color: rgba(255,255,255,0.70);
                }
                QLabel#fieldLabel {
                    font-size: 14px;
                    font-weight: 700;
                    color: #fff0e6;
                }
                QLineEdit {
                    padding: 14px 16px;
                    border: 1px solid rgba(255,255,255,0.14);
                    border-radius: 14px;
                    font-size: 14px;
                    background: rgba(255,255,255,0.08);
                    color: #fffaf5;
                    min-height: 22px;
                }
                QLineEdit:focus {
                    border: 1px solid #ff8d33;
                    background: rgba(255,255,255,0.12);
                }
                QLineEdit::placeholder {
                    color: rgba(255,255,255,0.45);
                }
                QFrame#editCard {
                    background: rgba(12,16,30,0.62);
                    border: 1px solid rgba(255,255,255,0.10);
                    border-radius: 24px;
                }
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #ff6a00, stop:1 #ff9533);
                    color: white;
                    border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 14px;
                    padding: 12px;
                    font-size: 15px;
                    font-weight: bold;
                    min-height: 18px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #ff7b14, stop:1 #ffa347);
                }
                QPushButton#saveBtn { }
                QPushButton#cancelBtn {
                    background: rgba(255,255,255,0.10);
                    color: #fff7ef;
                    border: 1px solid rgba(255,255,255,0.12);
                }
                QPushButton#cancelBtn:hover {
                    background: rgba(255,255,255,0.16);
                }
                QPushButton#toggleBtn {
                    background: #6c757d;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-size: 15px;
                    font-weight: bold;
                    padding: 0px;
                }
                QPushButton#toggleBtn:hover {
                    background: #5a6268;
                }
                QScrollArea {
                    background: transparent;
                    border: none;
                }
                QScrollBar:vertical {
                    border: none;
                    background: transparent;
                    width: 10px;
                    margin: 8px 4px 8px 4px;
                }
                QScrollBar::handle:vertical {
                    background: rgba(255,255,255,0.20);
                    min-height: 40px;
                    border-radius: 5px;
                }
                QScrollBar::handle:vertical:hover {
                    background: rgba(255,255,255,0.28);
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                }
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                    background: transparent;
                }
            """)

            outer = QVBoxLayout(dialog)
            outer.setContentsMargins(26, 26, 26, 26)
            outer.setSpacing(14)

            card = QFrame()
            card.setObjectName("editCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(34, 28, 34, 24)
            card_layout.setSpacing(12)
            outer.addWidget(card)

            header = QLabel(f"Edit {user_data.get('role', 'User')}")
            header.setObjectName("titleLabel")
            header.setAlignment(Qt.AlignCenter)

            card_layout.addWidget(header)

            info = QLabel(f"{user_data.get('role', '')} • {user_data.get('organization', '')}")
            info.setObjectName("subtitleLabel")
            info.setAlignment(Qt.AlignCenter)

            card_layout.addWidget(info)

            card_layout.addSpacing(4)

            fields = [
                ("Full Name", "full_name"),
                ("Age", "age"),
                ("Gender", "gender"),
                ("Address", "address"),
                ("Phone Number", "phone"),
                ("Email ID", "email"),
                ("Password", "password"),
                ("Confirm Password", "confirm_password"),
            ]

            def toggle_visibility(password_field, eye_button):
                if password_field.echoMode() == QLineEdit.Password:
                    password_field.setEchoMode(QLineEdit.Normal)
                    eye_button.setText("🔒")
                else:
                    password_field.setEchoMode(QLineEdit.Password)
                    eye_button.setText("👁")

            placeholders = {
                "full_name": "Enter full name",
                "age": "Enter age",
                "gender": "M / F / O",
                "address": "Enter address",
                "phone": "10-digit phone number",
                "email": "name@example.com",
                "password": "Enter password",
                "confirm_password": "Re-enter password",
            }

            def toggle_visibility_icon(password_field, eye_button):
                if password_field.echoMode() == QLineEdit.Password:
                    password_field.setEchoMode(QLineEdit.Normal)
                    eye_button.setText("\U0001F512")  # 🔒
                else:
                    password_field.setEchoMode(QLineEdit.Password)
                    eye_button.setText("\U0001F441")  # 👁

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            try:
                scroll.setFrameShape(QFrame.NoFrame)
                scroll.viewport().setStyleSheet("background: transparent;")
            except Exception:
                pass

            form_body = QWidget()
            form_body.setStyleSheet("background: transparent;")
            form_layout = QVBoxLayout(form_body)
            form_layout.setContentsMargins(0, 10, 0, 0)
            form_layout.setSpacing(12)
            scroll.setWidget(form_body)
            card_layout.addWidget(scroll, 1)

            widgets = {}
            for label_text, key in fields:
                label = QLabel(f"{label_text}:")
                label.setObjectName("fieldLabel")
                form_layout.addWidget(label)

                edit = QLineEdit()
                edit.setText(str(user_data.get(key, "")) if user_data.get(key, "") is not None else "")
                edit.setMinimumSize(400, 40)
                edit.setMaximumWidth(560)
                if key in placeholders:
                    edit.setPlaceholderText(placeholders[key])

                if key == "phone":
                    # Not QIntValidator: its C++ int ceiling rejects every
                    # keystroke of any number above 2147483647.
                    from utils.input_validation import apply_digit_only, PHONE_DIGITS
                    apply_digit_only(edit, PHONE_DIGITS)

                if key == "age":
                    edit.setValidator(QIntValidator(0, 150, dialog))
                    edit.setMaxLength(3)

                if key in ["password", "confirm_password"]:
                    edit.setEchoMode(QLineEdit.Password)
                    edit.returnPressed.connect(dialog.accept)
                    pwd_row = QHBoxLayout()
                    pwd_row.setSpacing(12)
                    pwd_row.addWidget(edit, 1)
                    toggle_btn = QPushButton("👁")
                    toggle_btn.setObjectName("toggleBtn")
                    toggle_btn.setFixedSize(40, 40)
                    toggle_btn.setText("\U0001F441")  # 👁
                    toggle_btn.clicked.connect(
                        lambda checked=False, pwd_field=edit, btn=toggle_btn: toggle_visibility_icon(pwd_field, btn)
                    )
                    pwd_row.addWidget(toggle_btn)
                    form_layout.addLayout(pwd_row)
                else:
                    edit.setClearButtonEnabled(True)
                    form_layout.addWidget(edit)

                widgets[key] = edit
                form_layout.addSpacing(6)

            if widgets.get("confirm_password") and widgets.get("password"):
                widgets["confirm_password"].setText(widgets["password"].text())

            button_row = QHBoxLayout()
            button_row.setSpacing(12)
            save_btn = QPushButton("Save Changes")
            save_btn.setObjectName("saveBtn")
            cancel_btn = QPushButton("Cancel")
            cancel_btn.setObjectName("cancelBtn")
            save_btn.clicked.connect(dialog.accept)
            cancel_btn.clicked.connect(dialog.reject)
            button_row.addWidget(save_btn)
            button_row.addWidget(cancel_btn)
            card_layout.addSpacing(8)
            card_layout.addLayout(button_row)

            if dialog.exec_() != QDialog.Accepted:
                return False

            updated = dict(user_data)
            updated["full_name"] = widgets["full_name"].text().strip()
            updated["age"] = widgets["age"].text().strip()
            updated["gender"] = widgets["gender"].text().strip()
            updated["address"] = widgets["address"].text().strip()
            updated["phone"] = widgets["phone"].text().strip()
            updated["email"] = widgets["email"].text().strip()
            updated_password = widgets["password"].text()
            updated_confirm = widgets["confirm_password"].text()

            if not updated["full_name"]:
                QMessageBox.warning(self, "Error", "Full name is required.")
                return False
            if not updated["email"]:
                QMessageBox.warning(self, "Error", "Email ID is required.")
                return False
            if "@" not in updated["email"] or updated["email"].startswith("@") or updated["email"].endswith("@"):
                QMessageBox.warning(self, "Error", "Please enter a valid Email ID (must contain '@').")
                return False
            if not updated["phone"] or (not updated["phone"].isdigit()) or len(updated["phone"]) != 10:
                QMessageBox.warning(self, "Error", "Phone number must be numbers only and exactly 10 digits.")
                return False
            if not updated_password:
                QMessageBox.warning(self, "Error", "Password is required.")
                return False
            if updated_password != updated_confirm:
                QMessageBox.warning(self, "Error", "Passwords do not match.")
                return False

            updated["password"] = updated_password

            new_username = updated["phone"] if updated["phone"] else original_username
            if not self._update_created_user_across_stores(original_username, new_username, updated):
                return False

            user_data.clear()
            user_data.update(updated)
            user_data["username"] = new_username
            return True

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to edit user: {e}")
            return False

    def _update_created_user_across_stores(self, old_username, new_username, updated_user):
        """Update a created user record across creator file, clinical_users.json, and root users.json."""
        try:
            creator_role = self.user_data.get('role', '')
            creator_name = self.user_data.get('full_name', '')

            main_users = load_main_users()
            for key, record in main_users.items():
                if key == old_username:
                    continue
                if not isinstance(record, dict):
                    continue
                if new_username and key == new_username:
                    QMessageBox.warning(self, "Error", "A user with this phone/username already exists.")
                    return False
                if updated_user.get("phone") and str(record.get("phone", "")).strip() == str(updated_user.get("phone", "")).strip():
                    QMessageBox.warning(self, "Error", "A user with this phone number already exists.")
                    return False
                if updated_user.get("full_name") and str(record.get("full_name", "")).strip().lower() == str(updated_user.get("full_name", "")).strip().lower():
                    QMessageBox.warning(self, "Error", "A user with this full name already exists.")
                    return False

            if creator_role in ['Doctor Head', 'HCP Head'] and creator_name:
                created_users = load_created_users(creator_name, creator_role)
                if old_username in created_users:
                    created_users.pop(old_username, None)
                created_users[new_username] = updated_user
                save_created_users(created_users, creator_name, creator_role)

            clinical_users_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clinical_users.json")
            if os.path.exists(clinical_users_file):
                with open(clinical_users_file, "r") as f:
                    clinical_users = json.load(f)
            else:
                clinical_users = {}
            if old_username in clinical_users:
                clinical_users.pop(old_username, None)
            clinical_users[new_username] = updated_user
            with open(clinical_users_file, "w") as f:
                json.dump(clinical_users, f, indent=2)

            if old_username in main_users:
                main_users.pop(old_username, None)
            main_users[new_username] = updated_user
            root_users_file = str(data_file("users.json"))
            with open(root_users_file, "w") as f:
                json.dump(main_users, f, indent=2)

            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update user storage: {e}")
            return False
    
    def start_main_application(self, logged_in_user_data):
        """Start the main dashboard for a created user in view-only mode"""
        try:
            # Import required modules
            import sys
            import os
            
            # Add src directory to path if not already there
            src_dir = os.path.dirname(os.path.abspath(__file__))
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            
            # Use the restricted wrapper dashboard for users created by Doctor/HCP heads.
            from dashboard.restricted_dashboard import RestrictedDashboard
            if RestrictedDashboard is None:
                QMessageBox.critical(self, "Error", "Failed to load Dashboard module.")
                return

            user_record = logged_in_user_data or {}
            username = (
                user_record.get('username')
                or user_record.get('phone')
                or user_record.get('full_name')
                or 'User'
            )
            
            # Keep a reference so the new window is not garbage-collected.
            self.main_dashboard = RestrictedDashboard(
                username=username,
                role=user_record.get('role'),
                user_details=user_record,
                return_on_sign_out=self,
                root_login_dialog=self.parent(),
                parent=self.parent(),
            )

            # Keep the organization dashboard available so sign-out can return here.
            self.hide()
            self.main_dashboard.show()
            self.main_dashboard.raise_()
            self.main_dashboard.activateWindow()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start main application: {e}")
        
    def refresh_recent_activity(self):
        """Refresh the recent activity section"""
        if hasattr(self, 'user_count_label') and self.user_count_label:
            self.user_count_label.setText(f"Users: {self.get_user_count()}")
        self._populate_recent_users()

    def delete_created_user(self, user_data):
        """Delete a created user from all relevant JSON stores."""
        try:
            creator_role = self.user_data.get('role', '')
            creator_name = self.user_data.get('full_name', '')
            current_org = self.user_data.get('organization', '')
            username = user_data.get('username') or user_data.get('phone') or ''

            confirm_name = user_data.get('full_name', username or 'this user')
            reply = QMessageBox.question(
                self,
                "Delete User",
                f"Delete '{confirm_name}' permanently?\n\nThis will remove all saved user data.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

            removed_any = False

            if creator_role in ['Doctor Head', 'HCP Head'] and creator_name and username:
                created_users = load_created_users(creator_name, creator_role)
                if username in created_users:
                    created_users.pop(username, None)
                    save_created_users(created_users, creator_name, creator_role)
                    removed_any = True

            clinical_users_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clinical_users.json")
            if os.path.exists(clinical_users_file):
                with open(clinical_users_file, "r") as f:
                    clinical_users = json.load(f)

                removed_from_clinical = False
                if username and username in clinical_users:
                    clinical_users.pop(username, None)
                    removed_from_clinical = True
                else:
                    for key, record in list(clinical_users.items()):
                        if not isinstance(record, dict):
                            continue
                        if (
                            record.get('phone') == user_data.get('phone') and
                            record.get('full_name') == user_data.get('full_name') and
                            record.get('organization') == current_org
                        ):
                            clinical_users.pop(key, None)
                            removed_from_clinical = True
                            break

                if removed_from_clinical:
                    with open(clinical_users_file, "w") as f:
                        json.dump(clinical_users, f, indent=2)
                    removed_any = True

            root_users_file = str(data_file("users.json"))
            if os.path.exists(root_users_file):
                with open(root_users_file, "r") as f:
                    all_users = json.load(f)

                removed_from_root = False
                if username and username in all_users:
                    all_users.pop(username, None)
                    removed_from_root = True
                else:
                    for key, record in list(all_users.items()):
                        if not isinstance(record, dict):
                            continue
                        if (
                            record.get('phone') == user_data.get('phone') and
                            record.get('full_name') == user_data.get('full_name') and
                            record.get('organization') == current_org
                        ):
                            all_users.pop(key, None)
                            removed_from_root = True
                            break

                if removed_from_root:
                    with open(root_users_file, "w") as f:
                        json.dump(all_users, f, indent=2)
                    removed_any = True

            if removed_any:
                self.refresh_recent_activity()
                QMessageBox.information(self, "Deleted", "User deleted successfully.")
            else:
                QMessageBox.warning(self, "Not Found", "User record was not found in storage.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete user: {e}")

    def _clear_layout(self, layout):
        """Remove all widgets/items from a layout safely"""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)
                child_layout.deleteLater()

    def _populate_recent_users(self):
        """Render recent users list in the dashboard panel"""
        if not hasattr(self, 'recent_users_layout') or self.recent_users_layout is None:
            return

        self._clear_layout(self.recent_users_layout)
        recent_users = self.get_recent_users()

        if recent_users:
            from PyQt5.QtWidgets import QScrollArea
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setMaximumHeight(300)
            scroll_area.setStyleSheet("""
                QScrollArea {
                    background: transparent;
                    border: none;
                    border-radius: 16px;
                }
                QScrollBar:vertical {
                    background: rgba(255,255,255,0.08);
                    width: 12px;
                    margin: 4px 0 4px 0;
                    border-radius: 6px;
                }
                QScrollBar::handle:vertical {
                    background: rgba(255, 122, 26, 0.9);
                    min-height: 34px;
                    border-radius: 6px;
                }
                QScrollBar::handle:vertical:hover {
                    background: rgba(255, 145, 54, 1.0);
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                }
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                    background: transparent;
                }
            """)

            cards_container = QWidget()
            cards_container.setMinimumWidth(460)
            cards_container.setStyleSheet("""
                QWidget {
                    background: transparent;
                    border: none;
                }
            """)
            cards_layout = QVBoxLayout(cards_container)
            cards_layout.setSpacing(14)
            cards_layout.setContentsMargins(4, 4, 28, 4)

            for user_data in recent_users:
                cards_layout.addWidget(self.create_recent_activity_card(user_data))

            cards_layout.addStretch()
            scroll_area.setWidget(cards_container)
            self.recent_users_layout.addWidget(scroll_area)
            return

        no_users_label = QLabel("No users created yet. Click 'Add Users' to get started.")
        no_users_label.setStyleSheet("""
            color: #d8d9df;
            font-style: italic;
            padding: 30px 20px;
            background: rgba(255,255,255,0.05);
            border: 1px dashed rgba(255,255,255,0.16);
            border-radius: 16px;
        """)
        no_users_label.setAlignment(Qt.AlignCenter)
        self.recent_users_layout.addWidget(no_users_label)
    
    def get_user_count(self):
        """Get count of users created by current Doctor Head/HCP Head in the same organization"""
        try:
            current_role = self.user_data.get('role', '')
            current_org = self.user_data.get('organization', '')
            creator_name = self.user_data.get('full_name', '')
            
            if current_role in ['Doctor Head', 'HCP Head']:
                # For Doctor Head/HCP Head, count ONLY users they created in their organization
                return get_all_created_users_count(creator_name, current_role, current_org)
            else:
                # For other roles, count all users in the organization from clinical_users.json
                import os
                CLINICAL_USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clinical_users.json")
                count = 0
                
                if os.path.exists(CLINICAL_USERS_FILE):
                    with open(CLINICAL_USERS_FILE, "r") as f:
                        clinical_users = json.load(f)
                        for username, user_data in clinical_users.items():
                            if isinstance(user_data, dict) and user_data.get('organization') == current_org:
                                count += 1
                
                return count
        except Exception:
            return 0


class UserLoginDialog(QDialog):
    """Dialog for user login with full name and password"""
    
    def __init__(self, parent, user_data):
        super().__init__(parent)
        self.user_data = user_data
        self.init_ui()

    def _clear_suppress_empty_warning(self, *_args):
        self._suppress_empty_credentials_warning = False
    
    def keyPressEvent(self, event):
        """Handle key press events to prevent Enter key from closing dialog"""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # Check if the focus widget is a QLineEdit that has returnPressed connected
            focus_widget = self.focusWidget()
            if isinstance(focus_widget, QLineEdit):
                # Let the QLineEdit handle the Enter key (for returnPressed signals)
                focus_widget.keyPressEvent(event)
                return
            # Otherwise, ignore Enter key to prevent accidental dialog closure
            event.ignore()
            return
        # Handle other keys normally
        super().keyPressEvent(event)
    
    def init_ui(self):
        """Initialize login dialog UI"""
        self.setWindowTitle("User Login")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setFixedSize(620, 520)
        self.setModal(True)
        
        # Set background
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #161a31, stop:0.55 #1d2444, stop:1 #101427);
            }
            QLabel {
                color: #f5f1ea;
                font-size: 15px;
                font-weight: 600;
            }
            QLineEdit {
                padding: 14px 16px;
                border: 1px solid rgba(255,255,255,0.14);
                border-radius: 14px;
                font-size: 14px;
                background: rgba(255,255,255,0.08);
                color: #fffaf5;
                min-height: 22px;
            }
            QLineEdit:focus {
                border: 1px solid #ff8d33;
                background: rgba(255,255,255,0.12);
            }
            QLineEdit::placeholder {
                color: rgba(255,255,255,0.45);
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff6a00, stop:1 #ff9533);
                color: white;
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 14px;
                padding: 12px;
                font-size: 15px;
                font-weight: bold;
                min-height: 18px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff7b14, stop:1 #ffa347);
            }
            QPushButton#cancel_btn {
                background: rgba(255,255,255,0.10);
                color: #fff7ef;
            }
            QPushButton#cancel_btn:hover {
                background: rgba(255,255,255,0.16);
            }
            QFrame#loginCard {
                background: rgba(12,16,30,0.62);
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 24px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(24, 24, 24, 24)

        card = QFrame()
        card.setObjectName("loginCard")
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(18)
        card_layout.setContentsMargins(32, 30, 32, 28)
        layout.addWidget(card)

        # Title
        title = QLabel("Login to Continue")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 30px; font-weight: bold; color: #fff8f2; margin-top: 6px; border: none; background: transparent;")
        card_layout.addWidget(title)
        
        # User info
        user_info = QLabel(f"Welcome, {self.user_data.get('full_name', 'User')}")
        user_info.setAlignment(Qt.AlignCenter)
        user_info.setStyleSheet("font-size: 15px; color: rgba(255,255,255,0.72); margin-bottom: 8px; border: none; background: transparent;")
        card_layout.addWidget(user_info)
        
        # Full Name input
        name_label = QLabel("Full Name:")
        name_label.setStyleSheet("font-size: 14px; font-weight: 700; color: #fff0e6; margin-top: 8px; border: none; background: transparent;")
        card_layout.addWidget(name_label)
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter your full name")
        self.name_input.setMinimumSize(400, 40)
        self.name_input.returnPressed.connect(self.validate_login)
        self._suppress_empty_credentials_warning = False
        self.name_input.textChanged.connect(self._clear_suppress_empty_warning)
        card_layout.addWidget(self.name_input)
        
        # Password input
        password_label = QLabel("Password:")
        password_label.setStyleSheet("font-size: 14px; font-weight: 700; color: #fff0e6; margin-top: 6px; border: none; background: transparent;")
        card_layout.addWidget(password_label)
        
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText("Enter your password")
        self.password_input.setMinimumSize(400, 40)
        self.password_input.returnPressed.connect(self.validate_login)
        self.password_input.textChanged.connect(self._clear_suppress_empty_warning)
        password_row = QHBoxLayout()
        password_row.setSpacing(12)
        password_row.addWidget(self.password_input)

        self.password_toggle_btn = QPushButton("👁")
        self.password_toggle_btn.setFixedSize(40, 40)
        self.password_toggle_btn.setStyleSheet("""
            QPushButton {
                background: #6c757d;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 15px;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton:hover {
                background: #5a6268;
            }
        """)
        self.password_toggle_btn.clicked.connect(
            lambda: self.toggle_password_visibility(self.password_input, self.password_toggle_btn)
        )
        password_row.addWidget(self.password_toggle_btn)
        card_layout.addLayout(password_row)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(14)
        
        login_btn = QPushButton("Login")
        login_btn.clicked.connect(self.validate_login)
        login_btn.setMinimumSize(120, 46)
        login_btn.setAutoDefault(True)
        login_btn.setDefault(True)
        buttons_layout.addWidget(login_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("cancel_btn")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setMinimumSize(120, 46)
        buttons_layout.addWidget(cancel_btn)
        
        card_layout.addSpacing(8)
        card_layout.addLayout(buttons_layout)
        
        # Set default values for testing (remove in production)
        self.name_input.setText(self.user_data.get('full_name', ''))
        self.password_input.setFocus()

    def toggle_password_visibility(self, password_field, eye_button):
        """Toggle password visibility between hidden and visible"""
        if password_field.echoMode() == QLineEdit.Password:
            password_field.setEchoMode(QLineEdit.Normal)
            eye_button.setText("🔒")
        else:
            password_field.setEchoMode(QLineEdit.Password)
            eye_button.setText("👁")
    
    def validate_login(self):
        """Validate login credentials"""
        entered_name = self.name_input.text().strip()
        entered_password = self.password_input.text().strip()
        
        if not entered_name or not entered_password:
            if getattr(self, "_suppress_empty_credentials_warning", False):
                self._suppress_empty_credentials_warning = False
                return
            QMessageBox.warning(self, "Error", "Please enter both full name and password.")
            return
        
        # Check if credentials match (simple validation)
        stored_name = self.user_data.get('full_name', '')
        stored_password = self.user_data.get('password', '')  # Assuming password is stored
        
        if entered_name == stored_name and entered_password == stored_password:
            self.accept()  # Login successful
        else:
            QMessageBox.warning(self, "Login Failed", "Invalid full name or password. Please try again.")
            self._suppress_empty_credentials_warning = True
            self.password_input.blockSignals(True)
            self.password_input.clear()
            self.password_input.blockSignals(False)
            self.password_input.setFocus()
    
    def get_user_count(self):
        """Get count of users created by current Doctor Head/HCP Head in the same organization"""
        try:
            current_role = self.user_data.get('role', '')
            current_org = self.user_data.get('organization', '')
            creator_name = self.user_data.get('full_name', '')
            
            if current_role in ['Doctor Head', 'HCP Head']:
                # For Doctor Head/HCP Head, count ONLY users they created in their organization
                return get_all_created_users_count(creator_name, current_role, current_org)
            else:
                # For other roles, count all users in the organization from clinical_users.json
                import os
                CLINICAL_USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clinical_users.json")
                count = 0
                
                if os.path.exists(CLINICAL_USERS_FILE):
                    with open(CLINICAL_USERS_FILE, "r") as f:
                        clinical_users = json.load(f)
                        for username, user_data in clinical_users.items():
                            if isinstance(user_data, dict) and user_data.get('organization') == current_org:
                                count += 1
                
                return count
        except Exception:
            return 0


class OrganizationRequestHandler:
    """Handles organization request workflow"""
    
    def __init__(self, parent_dialog):
        self.parent_dialog = parent_dialog
        self.org_manager = OrganizationManager()
    
    def handle_organization_request(self):
        """Handle request for new organization"""
        dialog = QDialog(self.parent_dialog)
        dialog.setWindowTitle("Request New Organization")
        dialog.setModal(True)
        dialog.setFixedSize(480, 270)
        dialog.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #171b31, stop:0.55 #1d2444, stop:1 #12162a);
            }
            QFrame#requestCard {
                background: rgba(12,16,30,0.62);
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 22px;
            }
            QLabel {
                color: #f6f1eb;
                font-size: 14px;
                font-weight: 600;
                background: transparent;
                border: none;
            }
            QLabel#requestTitle {
                font-size: 22px;
                font-weight: bold;
                color: #fff8f2;
            }
            QLabel#requestSubtitle {
                font-size: 13px;
                font-weight: 500;
                color: rgba(255,255,255,0.68);
            }
            QLineEdit {
                padding: 12px 14px;
                border: 1px solid rgba(255,255,255,0.14);
                border-radius: 14px;
                font-size: 14px;
                color: #fffaf5;
                background: rgba(255,255,255,0.08);
            }
            QLineEdit:focus {
                border: 1px solid #ff8d33;
                background: rgba(255,255,255,0.12);
            }
            QLineEdit::placeholder {
                color: rgba(255,255,255,0.42);
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff6a00, stop:1 #ff9533);
                color: white;
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 12px;
                padding: 10px 18px;
                font-size: 14px;
                font-weight: bold;
                min-width: 110px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff7b14, stop:1 #ffa347);
            }
            QPushButton#cancelBtn {
                background: rgba(255,255,255,0.10);
                color: #fff7ef;
            }
            QPushButton#cancelBtn:hover {
                background: rgba(255,255,255,0.16);
            }
        """)

        outer_layout = QVBoxLayout(dialog)
        outer_layout.setContentsMargins(20, 20, 20, 20)

        card = QFrame()
        card.setObjectName("requestCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(26, 24, 26, 24)
        card_layout.setSpacing(16)

        title = QLabel("Request New Organization")
        title.setObjectName("requestTitle")
        title.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(title)

        subtitle = QLabel("Enter the organization name below")
        subtitle.setObjectName("requestSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(subtitle)

        org_name_input = QLineEdit()
        org_name_input.setPlaceholderText("Organization name")
        org_name_input.setMinimumHeight(46)
        org_name_input.returnPressed.connect(dialog.accept)
        card_layout.addWidget(org_name_input)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        ok_btn = QPushButton("OK")
        ok_btn.setMinimumHeight(46)
        ok_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("cancelBtn")
        cancel_btn.setMinimumHeight(46)
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)

        card_layout.addLayout(button_layout)
        outer_layout.addWidget(card)

        org_name = org_name_input.text()
        ok = False
        org_name_input.setFocus()
        if dialog.exec_() == QDialog.Accepted:
            org_name = org_name_input.text()
            ok = True
        
        if ok and org_name.strip():
            # Add the organization
            success, message = self.org_manager.add_organization(org_name.strip())
            
            if success:
                QMessageBox.information(self.parent_dialog, "Success", f"'{org_name.strip()}' added successfully!")
            else:
                QMessageBox.warning(self.parent_dialog, "Error", message)
    
    def handle_existing_organization_request(self):
        """Handle request for existing organization name"""
        # Get list of existing organizations
        organizations = self.org_manager.load_organizations()
        
        if not organizations:
            QMessageBox.warning(self.parent_dialog, "No Organizations", 
                              "No existing organizations found. Please request a new organization first.")
            return
        
        # Create a dialog to select existing organization
        dialog = QDialog(self.parent_dialog)
        dialog.setWindowTitle("Select Existing Organization")
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.setMinimumSize(560, 460)
        dialog.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #141729, stop:0.55 #1b2140, stop:1 #11162c);
                color: #f7f2ee;
            }
            QLabel {
                color: #f7f2ee;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff6a00, stop:1 #ff8a1d);
                color: white;
                border: 1px solid #ff9a3d;
                border-radius: 14px;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: bold;
                min-height: 50px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff7a14, stop:1 #ffa13c);
                border: 1px solid #ffb45f;
            }
            QPushButton:pressed {
                background: #e65f00;
                padding-top: 13px;
            }
            QListWidget {
                background: rgba(255,255,255,0.07);
                border: 1px solid #ff7b1a;
                border-radius: 14px;
                padding: 10px;
                color: #f7f2ee;
                font-size: 14px;
                outline: none;
            }
            QListWidget::item {
                padding: 4px;
                margin: 4px 0;
                border: none;
                border-radius: 10px;
                min-height: 30px;
            }
            QListWidget::item:selected {
                background: transparent;
            }
            QListWidget::item:hover {
                background: transparent;
            }
            QWidget#orgItem {
                background: rgba(255,255,255,0.05);
                border: 1px solid rgba(255, 132, 39, 0.18);
                border-radius: 12px;
            }
            QWidget#orgItem[selected="true"] {
                background: rgba(255, 116, 18, 0.22);
                border: 1px solid #ff8f33;
            }
            QLabel#orgName {
                color: #fff7f0;
                font-size: 14px;
                font-weight: 700;
                letter-spacing: 0.3px;
                border: none;
            }
            QPushButton#deleteOrgButton {
                background: rgba(255, 122, 26, 0.96);
                border: 1px solid rgba(255, 188, 127, 0.45);
                border-radius: 17px;
                color: white;
                font-size: 14px;
                font-weight: bold;
                min-height: 34px;
                max-height: 34px;
                min-width: 34px;
                max-width: 34px;
                padding: 0px;
            }
            QPushButton#deleteOrgButton:hover {
                background: #ff4d4d;
                border: 1px solid #ff8a8a;
            }
            QPushButton#deleteOrgButton:pressed {
                background: #d83a3a;
            }
            QPushButton#cancelButton {
                background: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.16);
                color: #fff7f0;
            }
            QPushButton#cancelButton:hover {
                background: rgba(255,255,255,0.14);
                border: 1px solid rgba(255,255,255,0.25);
            }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 22, 20, 18)
        layout.setSpacing(14)
        
        # Title
        title = QLabel("Select Existing Organization")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; margin-bottom: 18px; color: #fff9f4;")
        layout.addWidget(title)
        
        # Organization list with delete functionality
        from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QWidget, QHBoxLayout
        from PyQt5.QtCore import QSize
        org_list = QListWidget()
        org_list.setMinimumHeight(180)
        org_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        org_list.setSpacing(8)
        
        for org_name in organizations.keys():
            # Create custom widget for each organization item
            item_widget = QWidget()
            item_widget.setObjectName("orgItem")
            item_widget.setProperty("selected", False)
            item_widget.setMinimumHeight(52)
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(14, 8, 14, 8)
            item_layout.setSpacing(12)
            
            # Organization name label
            org_label = QLabel(org_name)
            org_label.setObjectName("orgName")
            org_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            org_label.setMinimumWidth(0)
            org_label.setStyleSheet("min-height: 18px; border: none;")
            item_layout.addWidget(org_label)
            
            # Add stretch to push delete button to the right
            item_layout.addStretch()
            
            # Delete button
            delete_btn = QPushButton("✕")
            delete_btn.setObjectName("deleteOrgButton")
            delete_btn.setFixedSize(34, 34)
            delete_btn.setFocusPolicy(Qt.NoFocus)
            delete_btn.setText("✕")
            
            # Connect delete button to delete function
            def delete_organization(current_org_name, current_dialog):
                reply = QMessageBox.question(
                    current_dialog, 
                    "Confirm Delete", 
                    f"Are you sure you want to delete the organization '{current_org_name}'?\n\nThis will also remove all Doctor Head and HCP Head users associated with this organization.\n\nThis action cannot be undone.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    print(f"DEBUG: Attempting to delete organization '{current_org_name}'")
                    
                    # Remove associated Doctor Head users
                    try:
                        doctor_head_users = load_doctor_head_users()
                        print(f"DEBUG: Loaded Doctor Head users: {list(doctor_head_users.keys())}")
                        users_to_remove = []
                        for username, user_data in doctor_head_users.items():
                            print(f"DEBUG: Checking user {username}: org='{user_data.get('organization')}' vs '{current_org_name}'")
                            if user_data.get('organization') == current_org_name:
                                users_to_remove.append(username)
                        
                        print(f"DEBUG: Doctor Head users to remove: {users_to_remove}")
                        for username in users_to_remove:
                            doctor_head_users.pop(username, None)
                            print(f"DEBUG: Removed Doctor Head user {username} from organization {current_org_name}")
                        
                        if users_to_remove:
                            save_head_users(doctor_head_users, 'Doctor Head')
                            print(f"DEBUG: Removed {len(users_to_remove)} Doctor Head users for organization {current_org_name}")
                        else:
                            print(f"DEBUG: No Doctor Head users found for organization {current_org_name}")
                    except Exception as e:
                        print(f"DEBUG: Error removing Doctor Head users: {e}")
                    
                    # Remove associated HCP Head users
                    try:
                        hcp_head_users = load_hcp_head_users()
                        print(f"DEBUG: Loaded HCP Head users: {list(hcp_head_users.keys())}")
                        users_to_remove = []
                        for username, user_data in hcp_head_users.items():
                            print(f"DEBUG: Checking HCP user {username}: org='{user_data.get('organization')}' vs '{current_org_name}'")
                            if user_data.get('organization') == current_org_name:
                                users_to_remove.append(username)
                        
                        print(f"DEBUG: HCP Head users to remove: {users_to_remove}")
                        for username in users_to_remove:
                            hcp_head_users.pop(username, None)
                            print(f"DEBUG: Removed HCP Head user {username} from organization {current_org_name}")
                        
                        if users_to_remove:
                            save_head_users(hcp_head_users, 'HCP Head')
                            print(f"DEBUG: Removed {len(users_to_remove)} HCP Head users for organization {current_org_name}")
                        else:
                            print(f"DEBUG: No HCP Head users found for organization {current_org_name}")
                    except Exception as e:
                        print(f"DEBUG: Error removing HCP Head users: {e}")
                    
                    # Remove organization from the organizations dict
                    organizations.pop(current_org_name, None)
                    print(f"DEBUG: Organization removed from dict. Remaining: {list(organizations.keys())}")
                    
                    # Save updated organizations
                    if self.org_manager.save_organizations(organizations):
                        print(f"DEBUG: Organizations saved successfully")
                        QMessageBox.information(current_dialog, "Success", f"Organization '{current_org_name}' and all associated users deleted successfully.")
                        # Clean up orphaned users after organization deletion
                        cleanup_orphaned_users()
                        # Close and reopen the dialog to refresh the list
                        current_dialog.accept()
                        self.handle_existing_organization_request()
                        return
                    else:
                        print(f"DEBUG: Failed to save organizations")
                        QMessageBox.warning(current_dialog, "Error", "Failed to delete organization.")
            
            delete_btn.clicked.connect(lambda: delete_organization(org_name, dialog))
            item_layout.addWidget(delete_btn, 0, Qt.AlignVCenter)
            
            # Create list item and set the custom widget
            list_item = QListWidgetItem(org_list)
            list_item.setSizeHint(QSize(0, 56))
            org_list.setItemWidget(list_item, item_widget)

        def refresh_org_item_styles():
            current_item = org_list.currentItem()
            for index in range(org_list.count()):
                list_item = org_list.item(index)
                widget = org_list.itemWidget(list_item)
                if widget:
                    widget.setProperty("selected", list_item == current_item)
                    widget.style().unpolish(widget)
                    widget.style().polish(widget)
                    widget.update()

        org_list.currentItemChanged.connect(lambda current, previous: refresh_org_item_styles())
        refresh_org_item_styles()
        
        layout.addWidget(org_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        select_btn = QPushButton("Select")
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("cancelButton")
        
        def select_organization():
            current_item = org_list.currentItem()
            if current_item:
                # Get the widget from the list item to extract organization name
                widget = org_list.itemWidget(current_item)
                if widget:
                    # Robust extraction: get the org label by objectName instead of scanning children.
                    org_label = widget.findChild(QLabel, "orgName")
                    selected_org = (org_label.text().strip() if org_label else "")
                    if selected_org:
                        print(f"DEBUG: Selected organization: {selected_org}")
                        dialog.accept()
                        self.show_role_selection_dialog(selected_org, is_existing_organization=True)
                        return

                    # Find the organization label in the widget
                    # Look for the first QLabel that is not the delete button
                    for child in widget.children():
                        if isinstance(child, QLabel):
                            # Check if this is the organization label (not delete button)
                            # Delete button has "X" text, organization label has org name
                            child_text = child.text()
                            if child_text and child_text != "X" and child_text != "✕":
                                selected_org = child_text
                                print(f"DEBUG: Selected organization: {selected_org}")
                                dialog.accept()
                                # Show role selection dialog
                                self.show_role_selection_dialog(selected_org, is_existing_organization=True)
                                return
                    # If no valid label found, try to get from organization data
                    QMessageBox.warning(dialog, "Warning", "Could not extract organization name. Please try again.")
                else:
                    QMessageBox.warning(dialog, "Warning", "No widget found for selected item.")
            else:
                QMessageBox.warning(dialog, "Warning", "Please select an organization.")
        
        select_btn.clicked.connect(select_organization)
        cancel_btn.clicked.connect(dialog.reject)
        
        button_layout.addWidget(select_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        # Execute dialog
        dialog.exec_()
    
    def show_role_selection_dialog(self, organization_name, is_existing_organization=False):
        """Show role selection dialog with Doctor Head and HCP Head options"""
        dialog = RoleSelectionDialog(self, organization_name, is_existing_organization)
        
        if dialog.exec_() == QDialog.Accepted:
            role, org, is_login = dialog.get_selection()
            print(f"DEBUG: RoleSelectionDialog returned - role: {role}, org: {org}, is_login: {is_login}")
            print(f"DEBUG: hasattr login_user_data: {hasattr(dialog, 'login_user_data')}")
            print(f"DEBUG: hasattr signup_user_data: {hasattr(dialog, 'signup_user_data')}")
            if hasattr(dialog, 'login_user_data'):
                print(f"DEBUG: login_user_data: {dialog.login_user_data}")
            if hasattr(dialog, 'signup_user_data'):
                print(f"DEBUG: signup_user_data: {dialog.signup_user_data}")
            
            # Store the selection in the parent dialog
            self.parent_dialog.selected_role = role
            self.parent_dialog.selected_organization = org
            
            if is_login:
                # Handle login for existing user - check if login was completed
                if hasattr(dialog, 'login_user_data') and dialog.login_user_data:
                    # Login was successful, open dashboard
                    QMessageBox.information(self.parent_dialog, "Login Complete", 
                        f"Login successful!\nRole: {role}\nOrganization: {org}\n\nOpening your dashboard...")
                    
                    # Open the dashboard
                    self.open_dashboard(dialog.login_user_data)
                else:
                    # Login was cancelled or failed
                    QMessageBox.information(self.parent_dialog, "Login Cancelled", 
                        "Login was cancelled or incomplete.")
            else:
                # Handle signup for new user - check if signup was completed
                if hasattr(dialog, 'signup_user_data') and dialog.signup_user_data:
                    # Sign-up was successful, open dashboard
                    QMessageBox.information(self.parent_dialog, "Registration Complete", 
                        f"Registration successful!\nRole: {role}\nOrganization: {org}\n\nOpening your dashboard...")
                    
                    # Open the dashboard
                    self.open_dashboard(dialog.signup_user_data)
                else:
                    # Sign-up was cancelled or failed
                    QMessageBox.information(self.parent_dialog, "Registration Cancelled", 
                        "Registration was cancelled or incomplete.")
            
            return role, org, is_login
        
        return None, None, None
    
    def open_dashboard(self, user_data):
        """Open the dashboard window with user data"""
        try:
            dashboard = DashboardWindow(self.parent_dialog, user_data)
            dashboard.show()
            # Bring dashboard to front
            dashboard.raise_()
            dashboard.activateWindow()
        except Exception as e:
            QMessageBox.critical(self.parent_dialog, "Error", f"Failed to open dashboard: {str(e)}")


def create_organization_request_button(parent_dialog):
    """Create and return the organization request button with connected handler"""
    org_request_btn = QPushButton("Request for New Organization")
    org_request_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    org_request_btn.setMinimumHeight(32)
    org_request_btn.setStyleSheet("""
        QPushButton {
            background: transparent;
            color: #ff6600;
            border: none;
            border-radius: 8px;
            padding: 6px 10px;
            font-size: 14px;
            font-weight: bold;
            text-decoration: underline;
            min-height: 32px;
            text-align: center;
        }
        QPushButton:hover {
            background: rgba(255, 102, 0, 0.1);
        }
    """)
    
    # Create handler and connect button
    handler = OrganizationRequestHandler(parent_dialog)
    org_request_btn.clicked.connect(handler.handle_organization_request)
    
    return org_request_btn, handler


def create_existing_organization_button(parent_dialog):
    """Create and return the existing organization button with connected handler"""
    existing_org_btn = QPushButton("Existing Organization")
    existing_org_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    existing_org_btn.setMinimumHeight(32)
    existing_org_btn.setStyleSheet("""
        QPushButton {
            background: transparent;
            color: #ff6600;
            border: none;
            border-radius: 8px;
            padding: 6px 10px;
            font-size: 14px;
            font-weight: bold;
            text-decoration: underline;
            min-height: 32px;
            text-align: center;
        }
        QPushButton:hover {
            background: rgba(255, 102, 0, 0.1);
        }
    """)
    
    # Create handler and connect button
    handler = OrganizationRequestHandler(parent_dialog)
    existing_org_btn.clicked.connect(handler.handle_existing_organization_request)
    
    return existing_org_btn, handler


def create_organization_buttons_layout(parent_dialog):
    """Create a layout with both organization buttons and return the layout and handlers"""
    from PyQt5.QtWidgets import QVBoxLayout
    
    # Create buttons layout (vertical)
    buttons_layout = QVBoxLayout()
    buttons_layout.setContentsMargins(0, 0, 0, 0)
    buttons_layout.setSpacing(6)
    
    # Create both buttons
    new_org_btn, new_handler = create_organization_request_button(parent_dialog)
    existing_org_btn, existing_handler = create_existing_organization_button(parent_dialog)
    
    # Add buttons to layout vertically
    buttons_layout.addWidget(new_org_btn, 0, Qt.AlignHCenter)
    buttons_layout.addWidget(existing_org_btn, 0, Qt.AlignHCenter)
    
    return buttons_layout, new_handler, existing_handler


class AddUsersDialog(QDialog):
    """Dialog for adding users based on role hierarchy"""
    
    def __init__(self, parent, current_user_data):
        super().__init__(parent)
        self.current_user_data = current_user_data
        self.current_role = current_user_data.get('role', '')
        self.init_ui()
    
    def keyPressEvent(self, event):
        """Handle key press events to prevent Enter key from closing dialog"""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # Check if the focus widget is a QLineEdit that has returnPressed connected
            focus_widget = self.focusWidget()
            if isinstance(focus_widget, QLineEdit):
                # Let the QLineEdit handle the Enter key (for returnPressed signals)
                focus_widget.keyPressEvent(event)
                return
            # Otherwise, ignore Enter key to prevent accidental dialog closure
            event.ignore()
            return
        # Handle other keys normally
        super().keyPressEvent(event)
    
    def init_ui(self):
        """Initialize the add users dialog UI"""
        self.setWindowTitle("Add Users")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setFixedSize(760, 520)
        self.setModal(True)
        
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #161a31, stop:0.55 #1d2444, stop:1 #101427);
            }
            QLabel {
                color: #f5f1ea;
                font-size: 15px;
                font-weight: 600;
                background: transparent;
                border: none;
            }
            QLabel#titleLabel {
                font-size: 30px;
                font-weight: bold;
                color: #fff8f2;
            }
            QLabel#subtitleLabel {
                font-size: 15px;
                font-weight: 600;
                color: rgba(255,255,255,0.72);
            }
            QLabel#sectionLabel {
                font-size: 15px;
                font-weight: 700;
                color: #fff0e6;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff6a00, stop:1 #ff9533);
                color: white;
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 14px;
                padding: 12px;
                font-size: 15px;
                font-weight: bold;
                min-height: 18px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff7b14, stop:1 #ffa347);
            }
            QFrame#addUsersCard {
                background: rgba(12,16,30,0.62);
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 24px;
            }
            QFrame#dividerLine {
                background: rgba(255,255,255,0.10);
            }
        """)
        
        outer = QVBoxLayout(self)
        outer.setSpacing(0)
        outer.setContentsMargins(24, 24, 24, 24)

        card = QFrame()
        card.setObjectName("addUsersCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(32, 30, 32, 28)
        card_layout.setSpacing(12)
        outer.addWidget(card)
        
        # Title
        title = QLabel(f"Add Users - {self.current_role}")
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("titleLabel")
        card_layout.addWidget(title)

        subtitle = QLabel("Select user type to add:")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setObjectName("subtitleLabel")
        card_layout.addWidget(subtitle)

        divider = QFrame()
        divider.setObjectName("dividerLine")
        divider.setFixedHeight(1)
        card_layout.addWidget(divider)

        def _mk_role_btn(text, role):
            btn = QPushButton(text)
            btn.setMinimumHeight(52)
            btn.setMinimumWidth(280)
            btn.clicked.connect(lambda checked=False, r=role: self.show_user_form(r))
            return btn
        
        # Role selection based on current user
        if self.current_role == "Doctor Head":
            # Clinical User buttons
            clinical_layout = QHBoxLayout()
            clinical_layout.setSpacing(16)
            clinical_layout.addStretch()
            clinical_layout.addWidget(_mk_role_btn("Add Sr. Clinical Doctor", "Sr. Clinical Doctor"))
            clinical_layout.addWidget(_mk_role_btn("Add Jr. Clinical Doctor", "Jr. Clinical Doctor"))
            clinical_layout.addStretch()
            card_layout.addLayout(clinical_layout)
            
        elif self.current_role == "HCP Head":
            # Admin User buttons
            admin_layout = QHBoxLayout()
            admin_layout.setSpacing(16)
            admin_layout.addStretch()
            admin_layout.addWidget(_mk_role_btn("Add Sr. Admin", "Sr. Admin"))
            admin_layout.addWidget(_mk_role_btn("Add Jr. Admin", "Jr. Admin"))
            admin_layout.addStretch()
            card_layout.addLayout(admin_layout)

            card_layout.addSpacing(18)
            
            # Sub Dealer buttons
            subtitle2 = QLabel("Or add Sub Dealer roles:")
            subtitle2.setAlignment(Qt.AlignCenter)
            subtitle2.setObjectName("subtitleLabel")
            card_layout.addWidget(subtitle2)

            sub_dealer_layout = QHBoxLayout()
            sub_dealer_layout.setSpacing(16)
            sub_dealer_layout.addStretch()
            sub_dealer_layout.addWidget(_mk_role_btn("Add Employee", "Employee"))
            sub_dealer_layout.addWidget(_mk_role_btn("Add Receptionist", "Receptionist"))
            sub_dealer_layout.addStretch()
            card_layout.addLayout(sub_dealer_layout)

        card_layout.addStretch()
    
    def show_user_form(self, user_role):
        """Show user creation form for the selected role"""
        form_dialog = UserCreationDialog(self, user_role, self.current_user_data)
        form_dialog.exec_()
    
    def get_user_data(self):
        """Return the created user data"""
        return self.user_data if hasattr(self, 'user_data') else None


class UserCreationDialog(QDialog):
    """Dialog for creating new users"""
    
    def __init__(self, parent, user_role, current_user_data):
        super().__init__(parent)
        self.user_role = user_role
        self.current_user_data = current_user_data
        self.user_data = {}
        self.return_to_dashboard_requested = False
        self._success_popup_shown = False
        self._create_in_progress = False
        self.init_ui()
    
    def keyPressEvent(self, event):
        """Handle key press events to prevent Enter key from closing dialog"""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # Check if the focus widget is a QLineEdit that has returnPressed connected
            focus_widget = self.focusWidget()
            if isinstance(focus_widget, QLineEdit):
                # Let the QLineEdit handle the Enter key (for returnPressed signals)
                focus_widget.keyPressEvent(event)
                return
            # Otherwise, ignore Enter key to prevent accidental dialog closure
            event.ignore()
            return
        # Handle other keys normally
        super().keyPressEvent(event)
    
    def init_ui(self):
        """Initialize the user creation dialog UI"""
        self.setWindowTitle(f"Create {self.user_role}")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setFixedSize(760, 780)
        self.setModal(True)
        
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #161a31, stop:0.55 #1d2444, stop:1 #101427);
            }
            QLabel {
                color: #f5f1ea;
                font-size: 15px;
                font-weight: 600;
                background: transparent;
                border: none;
            }
            QLabel#titleLabel {
                font-size: 30px;
                font-weight: bold;
                color: #fff8f2;
            }
            QLabel#subtitleLabel {
                font-size: 15px;
                font-weight: 600;
                color: rgba(255,255,255,0.72);
            }
            QLabel#fieldLabel {
                font-size: 15px;
                font-weight: 700;
                color: #fff0e6;
            }
            QLineEdit {
                padding: 14px 16px;
                border: 1px solid rgba(255,255,255,0.14);
                border-radius: 14px;
                font-size: 14px;
                background: rgba(255,255,255,0.08);
                color: #fffaf5;
                min-height: 22px;
            }
            QLineEdit:focus {
                border: 1px solid #ff8d33;
                background: rgba(255,255,255,0.12);
            }
            QLineEdit::placeholder {
                color: rgba(255,255,255,0.45);
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff6a00, stop:1 #ff9533);
                color: white;
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 14px;
                padding: 12px;
                font-size: 15px;
                font-weight: bold;
                min-height: 18px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff7b14, stop:1 #ffa347);
            }
            QPushButton#toggleBtn {
                background: #6c757d;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 15px;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton#toggleBtn:hover {
                background: #5a6268;
            }
            QPushButton#linkBtn {
                background: transparent;
                color: #ff8d33;
                border: none;
                text-decoration: underline;
                font-size: 12px;
                padding: 6px;
                min-height: 0px;
            }
            QPushButton#linkBtn:hover {
                color: #ffa347;
                background: rgba(255, 141, 51, 0.12);
            }
            QFrame#formCard {
                background: rgba(12,16,30,0.62);
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 24px;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                border: none;
                background: transparent;
                width: 10px;
                margin: 8px 4px 8px 4px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255,255,255,0.20);
                min-height: 40px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255,255,255,0.28);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
        """)
        
        outer = QVBoxLayout(self)
        outer.setSpacing(0)
        outer.setContentsMargins(24, 24, 24, 24)

        card = QFrame()
        card.setObjectName("formCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(32, 30, 32, 28)
        card_layout.setSpacing(12)
        outer.addWidget(card)
        
        # Title
        title = QLabel(f"Create {self.user_role}")
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("titleLabel")
        card_layout.addWidget(title)

        subtitle = QLabel("Fill in the details to create a new user")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setObjectName("subtitleLabel")
        card_layout.addWidget(subtitle)

        # Keep legacy code below working without changing logic flow:
        # use `layout` as an alias to the card layout.
        layout = card_layout
        
        # Form fields
        fields = [
            ("Full Name:", "full_name", "Enter full name"),
            ("Age:", "age", "Enter age"),
            ("Gender:", "gender", "Enter gender"),
            ("Address:", "address", "Enter address"),
            ("Phone Number:", "phone", "Enter phone number"),
            ("Email ID:", "email", "Enter email ID"),
            ("Password:", "password", "Enter password"),
            ("Confirm Password:", "confirm_password", "Confirm password")
        ]
        
        self.field_widgets = {}

        def _toggle_visibility(pwd_field, btn):
            try:
                if pwd_field.echoMode() == QLineEdit.Password:
                    pwd_field.setEchoMode(QLineEdit.Normal)
                    btn.setText("\U0001F512")  # 🔒
                else:
                    pwd_field.setEchoMode(QLineEdit.Password)
                    btn.setText("\U0001F441")  # 👁
            except Exception:
                pass
        
        for label_text, field_name, placeholder in fields:
            field_layout = QHBoxLayout()
            label = QLabel(label_text)
            label.setObjectName("fieldLabel")
            label.setFixedWidth(170)
            
            if field_name in ['password', 'confirm_password']:
                field = QLineEdit()
                field.setPlaceholderText(placeholder)
                field.setEchoMode(QLineEdit.Password)
                field.setMinimumSize(400, 40)
                field.setMaximumWidth(560)
                field.returnPressed.connect(self.handle_create_user)
                toggle_btn = QPushButton("👁")
                toggle_btn.setFixedSize(40, 40)
                toggle_btn.setObjectName("toggleBtn")
                toggle_btn.setText("\U0001F441")  # 👁
                toggle_btn.clicked.connect(lambda checked=False, pwd_field=field, btn=toggle_btn: _toggle_visibility(pwd_field, btn))
            else:
                field = QLineEdit()
                field.setPlaceholderText(placeholder)
                field.setMinimumSize(400, 40)
                field.setMaximumWidth(560)
                field.setClearButtonEnabled(True)
                if field_name == 'confirm_password':
                    field.returnPressed.connect(self.handle_create_user)
                if field_name == 'phone':
                    # Not QIntValidator: its C++ int ceiling rejects every
                    # keystroke of any number above 2147483647.
                    from utils.input_validation import apply_digit_only, PHONE_DIGITS
                    apply_digit_only(field, PHONE_DIGITS)
                if field_name == 'age':
                    field.setValidator(QIntValidator(0, 150, self))
                    field.setMaxLength(3)
            
            self.field_widgets[field_name] = field
            field_layout.addWidget(label)
            field_layout.addWidget(field)
            if field_name in ['password', 'confirm_password']:
                field_layout.addWidget(toggle_btn)
            field_layout.addStretch()
            layout.addLayout(field_layout)
            layout.addSpacing(8)
        
        # Create button
        create_btn = QPushButton("Create User")
        create_btn.setObjectName("create_btn")
        create_btn.clicked.connect(self.handle_create_user)
        layout.addWidget(create_btn)
        
        # Existing User link (hide for sub-users created by Doctor/HCP Head)
        creator_role = ""
        try:
            creator_role = (self.current_user_data or {}).get("role", "") or ""
        except Exception:
            creator_role = ""

        if creator_role not in ["Doctor Head", "HCP Head"]:
            existing_user_link = QPushButton("Existing User")
            existing_user_link.setObjectName("linkBtn")
            existing_user_link.clicked.connect(self.handle_existing_user)
            layout.addWidget(existing_user_link, 0, Qt.AlignHCenter)
        
        layout.addStretch()

    def toggle_password_visibility(self, password_field, eye_button):
        """Toggle password visibility between hidden and visible"""
        if password_field.echoMode() == QLineEdit.Password:
            password_field.setEchoMode(QLineEdit.Normal)
            eye_button.setText("🔒")
        else:
            password_field.setEchoMode(QLineEdit.Password)
            eye_button.setText("👁")
    
    def handle_create_user(self):
        """Handle user creation"""
        if self._create_in_progress or self._success_popup_shown:
            return

        # Get form data
        full_name = self.field_widgets['full_name'].text().strip()
        age = self.field_widgets['age'].text().strip()
        gender = self.field_widgets['gender'].text().strip()
        address = self.field_widgets['address'].text().strip()
        phone = self.field_widgets['phone'].text().strip()
        email = self.field_widgets['email'].text().strip()
        password = self.field_widgets['password'].text()
        confirm_password = self.field_widgets['confirm_password'].text()
        
        # Validation
        if not full_name or not password:
            QMessageBox.warning(self, "Error", "Full name and password are required.")
            return

        if not email:
            QMessageBox.warning(self, "Error", "Email ID is required.")
            return

        if "@" not in email or email.startswith("@") or email.endswith("@"):
            QMessageBox.warning(self, "Error", "Please enter a valid Email ID (must contain '@').")
            return

        if not phone or (not phone.isdigit()) or len(phone) != 10:
            QMessageBox.warning(self, "Error", "Phone number must be numbers only and exactly 10 digits.")
            return
        
        if password != confirm_password:
            QMessageBox.warning(self, "Error", "Passwords do not match.")
            return

        # Only mark as "in progress" after validations pass; otherwise the button can appear stuck.
        self._create_in_progress = True
        
        try:
            # Import user management functions - move import inside function to avoid circular import
            import importlib
            import __main__ as main_module

            # In the frozen executable, importing `main` can fail (entrypoint is `__main__`).
            if not hasattr(main_module, "load_users") or not hasattr(main_module, "save_users"):
                try:
                    main_module = importlib.import_module("main")
                except Exception:
                    pass

            load_users = getattr(main_module, "load_users", None)
            save_users = getattr(main_module, "save_users", None)
            
            # Use separate file for clinical users
            import os
            CLINICAL_USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clinical_users.json")
            
            # Load existing clinical users or create new dict
            if os.path.exists(CLINICAL_USERS_FILE):
                with open(CLINICAL_USERS_FILE, "r") as f:
                    clinical_users = json.load(f)
            else:
                clinical_users = {}
            
            # Generate username from phone or use full name
            username = phone if phone else full_name.replace(' ', '_').lower()

            # Ensure the created sub-user can log in from the main application login as well.
            # Because the main login can accept full name, enforce uniqueness of phone + full name in root users.json.
            main_users = load_main_users()
            if username in main_users:
                QMessageBox.warning(self, "Error", "A user with this phone/username already exists.")
                return

            input_full_name_norm = full_name.strip().lower()
            input_phone_norm = str(phone).strip()
            for _, record in main_users.items():
                if not isinstance(record, dict):
                    continue
                existing_phone = str(record.get("phone", "")).strip()
                existing_full_name = str(record.get("full_name", "")).strip().lower()
                if input_phone_norm and existing_phone and input_phone_norm == existing_phone:
                    QMessageBox.warning(self, "Error", "A user with this phone number already exists.")
                    return
                if input_full_name_norm and existing_full_name and input_full_name_norm == existing_full_name:
                    QMessageBox.warning(self, "Error", "A user with this full name already exists.")
                    return
            
            # Create new user record
            new_user = {
                'full_name': full_name,
                'age': age,
                'gender': gender,
                'address': address,
                'phone': phone,
                'email': email,
                'password': password,
                'role': self.user_role,
                'organization': self.current_user_data.get('organization', ''),
                'signup_date': f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            }
            
            # Only add created_by for clinical users, not regular users
            if self.user_role in ['Sr. Clinical Doctor', 'Jr. Clinical Doctor', 'Sr. Admin', 'Jr. Admin', 'Employee', 'Receptionist']:
                new_user['created_by'] = self.current_user_data.get('full_name', '')
            
            # Check if creator is Doctor Head/HCP Head and save to creator-specific file
            creator_role = self.current_user_data.get('role', '')
            creator_name = self.current_user_data.get('full_name', '')
            
            if creator_role in ['Doctor Head', 'HCP Head']:
                # Load existing users created by this Doctor Head/HCP Head
                created_users = load_created_users(creator_name, creator_role)
                created_users[username] = new_user
                
                # Save to creator-specific file
                if save_created_users(created_users, creator_name, creator_role):
                    print(f"DEBUG: User {username} saved to creator-specific file for {creator_name} ({creator_role})")
                else:
                    print(f"DEBUG: Failed to save user to creator-specific file")
            
            # Also save to clinical_users.json for clinical users (existing behavior)
            if self.user_role in ['Sr. Clinical Doctor', 'Jr. Clinical Doctor', 'Sr. Admin', 'Jr. Admin', 'Employee', 'Receptionist']:
                clinical_users[username] = new_user
                
                # Save clinical users
                with open(CLINICAL_USERS_FILE, "w") as f:
                    json.dump(clinical_users, f, indent=2)
                
                print(f"DEBUG: Clinical user {username} saved to clinical_users.json")

            # Save to root users.json so this created user can use normal login too.
            if not save_user_to_main_users(new_user, username):
                QMessageBox.warning(self, "Warning", "User created, but failed to save to main users.json for normal login.")
            
            self.user_data = new_user

            # Show success once only (prevents duplicate popups when returning focus).
            if not self._success_popup_shown:
                self._success_popup_shown = True
                QMessageBox.information(self, "Success", f"{self.user_role} created successfully!")

            # Hide this dialog so it doesn't remain visible behind the management dashboard.
            try:
                self.hide()
            except Exception:
                pass

            # Open user management dashboard (modal on the AddUsersDialog instead of this dialog).
            self.open_user_management_dashboard()

            # Close this creation dialog after management dashboard is handled.
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create user: {str(e)}")
        finally:
            self._create_in_progress = False
    
    def handle_existing_user(self):
        """Handle existing user link - close all dialogs and go to login"""
        # Close all dialogs and go to login
        self.parent().parent().close()  # Close AddUsersDialog
        self.parent().close()  # Close DashboardWindow
        # The main application will return to login screen
    
    def open_user_management_dashboard(self):
        """Open user management dashboard with user count"""
        parent_dialog = self.parent()
        user_mgmt_dialog = UserManagementDashboard(parent_dialog if parent_dialog is not None else self, self.current_user_data)
        if user_mgmt_dialog.exec_() == QDialog.Accepted and user_mgmt_dialog.return_to_dashboard_requested:
            self.return_to_dashboard_requested = True
            if parent_dialog is not None:
                try:
                    parent_dialog.accept()
                except Exception:
                    pass
    
    def get_user_data(self):
        """Return the created user data"""
        return self.user_data if hasattr(self, 'user_data') else None


class UserManagementDashboard(QDialog):
    """Dashboard for managing users with count display"""
    
    def __init__(self, parent, current_user_data):
        super().__init__(parent)
        self.current_user_data = current_user_data
        self.return_to_dashboard_requested = False
        self.init_ui()
    
    def keyPressEvent(self, event):
        """Handle key press events to prevent Enter key from closing dialog"""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # Check if the focus widget is a QLineEdit that has returnPressed connected
            focus_widget = self.focusWidget()
            if isinstance(focus_widget, QLineEdit):
                # Let the QLineEdit handle the Enter key (for returnPressed signals)
                focus_widget.keyPressEvent(event)
                return
            # Otherwise, ignore Enter key to prevent accidental dialog closure
            event.ignore()
            return
        # Handle other keys normally
        super().keyPressEvent(event)
    
    def init_ui(self):
        """Initialize user management dashboard UI"""
        self.setWindowTitle("User Management")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setMinimumSize(900, 600)
        self.setModal(True)
        
        # Set background gradient
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #1a1a2e, stop:1 #16213e);
            }
            QLabel {
                color: white;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton {
                background: #ff6600;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #ff8800;
            }
            QFrame {
                background: rgba(255,255,255,0.1);
                border-radius: 12px;
                border: 1px solid rgba(255,255,255,0.2);
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        header_label = QLabel("User Management Dashboard")
        header_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(header_label)
        
        # User count section
        count_frame = QFrame()
        count_layout = QVBoxLayout(count_frame)
        count_layout.setContentsMargins(20, 20, 20, 20)
        
        # Calculate user count
        user_count = self.get_user_count()
        
        count_label = QLabel(f"Total Users: {user_count}")
        count_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #28a745; margin-bottom: 10px;")
        count_label.setAlignment(Qt.AlignCenter)
        count_layout.addWidget(count_label)
        
        subtitle_label = QLabel("Users created under your organization")
        subtitle_label.setStyleSheet("font-size: 14px; color: #ccc;")
        subtitle_label.setAlignment(Qt.AlignCenter)
        count_layout.addWidget(subtitle_label)
        
        layout.addWidget(count_frame)
        
        # Action buttons
        buttons_frame = QFrame()
        buttons_layout = QHBoxLayout(buttons_frame)
        
        back_btn = QPushButton("Back to Dashboard")
        back_btn.clicked.connect(self.back_to_dashboard)
        buttons_layout.addWidget(back_btn)
        
        add_more_btn = QPushButton("Add More Users")
        add_more_btn.clicked.connect(self.add_more_users)
        buttons_layout.addWidget(add_more_btn)

        layout.addWidget(buttons_frame)
        layout.addStretch()

    def get_user_count(self):
        """Get count of users created by current Doctor Head/HCP Head in the same organization"""
        try:
            current_role = self.current_user_data.get('role', '')
            current_org = self.current_user_data.get('organization', '')
            creator_name = self.current_user_data.get('full_name', '')
            
            if current_role in ['Doctor Head', 'HCP Head']:
                # For Doctor Head/HCP Head, count ONLY users they created in their organization
                return get_all_created_users_count(creator_name, current_role, current_org)
            else:
                # For other roles, count all users in the organization from clinical_users.json
                import os
                CLINICAL_USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clinical_users.json")
                count = 0
                
                if os.path.exists(CLINICAL_USERS_FILE):
                    with open(CLINICAL_USERS_FILE, "r") as f:
                        clinical_users = json.load(f)
                        for username, user_data in clinical_users.items():
                            if isinstance(user_data, dict) and user_data.get('organization') == current_org:
                                count += 1
                
                return count
        except Exception:
            return 0

    def add_more_users(self):
        """Return to existing Add Users dialog"""
        self.accept()

    def back_to_dashboard(self):
        """Return to the main dashboard"""
        self.return_to_dashboard_requested = True
        self.accept()
