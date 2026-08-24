"""
Shared input limits and validators for CardioX forms — src/utils/input_validation.py
====================================================================================

Single source of truth for "what counts as an acceptable value" in every form the
application shows: login, signup, organisation details, and the mobile-number
lookup in waveform analysis.

WHY A SHARED MODULE
-------------------
The same field was previously constrained in three different ways in three files
— one form capped the mobile number's length but accepted letters, another used
an integer validator whose ceiling silently rejected most real mobile numbers,
and a third checked nothing at all. Limits that live in one place cannot drift
apart like that, and they can be unit-tested without a display.

TWO LAYERS, ON PURPOSE
----------------------
Every rule is enforced twice:

  1. At the widget, via `apply_digit_only()` / `apply_text_limits()`, so bad
     input cannot be typed or pasted in the first place. This is a usability
     layer — it is trivially bypassed by anything that is not the GUI.
  2. At the logic layer, via the `validate_*` functions, before the value is
     stored, sent to the cloud, or used to build a request. This is the layer
     that actually holds, because it runs no matter how the value arrived.

Never rely on layer 1 alone. A widget validator is a convenience for the person
typing, not a security control.

NO Qt AT IMPORT TIME
--------------------
PyQt5 is imported lazily inside the two widget helpers. Everything else in this
module is pure Python, so the full rule set can be tested headlessly and reused
from non-GUI code (cloud upload, report generation, CLI tooling).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════════
# STANDARD FIELD LIMITS
# ══════════════════════════════════════════════════════════════════════════════
# Upper bounds are deliberately generous for humans and firmly finite for
# machines: a bounded field cannot be used to push a multi-megabyte string into
# users.json, a PDF, a log line, or a cloud payload.

PHONE_DIGITS: int = 10               # exactly ten digits, no country code

PASSWORD_MIN_LENGTH: int = 8
PASSWORD_MAX_LENGTH: int = 128       # bounded so a huge input cannot stall PBKDF2

USERNAME_MIN_LENGTH: int = 3
USERNAME_MAX_LENGTH: int = 64

NAME_MIN_LENGTH: int = 2
NAME_MAX_LENGTH: int = 100

EMAIL_MAX_LENGTH: int = 254          # RFC 5321 maximum path length

ORG_NAME_MAX_LENGTH: int = 120
ADDRESS_MAX_LENGTH: int = 200
SERIAL_MAX_LENGTH: int = 64
NOTES_MAX_LENGTH: int = 500

AGE_MIN: int = 0
AGE_MAX: int = 120

# Login attempts allowed before the form locks, and for how long. Mirrors the
# OTP path, which already locked after 3 tries for 5 minutes while the password
# path counted nothing at all.
LOGIN_MAX_ATTEMPTS: int = 5
LOGIN_LOCKOUT_SECONDS: int = 300


# ══════════════════════════════════════════════════════════════════════════════
# PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

# A DENY-list, not an allow-list — and deliberately so.
#
# An allow-list of [A-Za-z0-9 ...] looks tidier but rejects "Dr. José García",
# "Zoë Müller" and "दिव्यांश शर्मा". This is a clinical application used in
# India; refusing a doctor's own name is not a security posture, it is a defect.
#
# So Unicode letters from any script are accepted, and only the characters that
# genuinely change meaning at one of this value's sinks are refused:
#   <  >        reportlab Paragraph parses inline markup, and any HTML sink
#   {  }        template / format-string interpolation
#   [  ]  \     escaping and path construction
#   |  `  $  ;  shell metacharacters, should a value ever reach a command line
#   "           quoting
# Control characters are already gone by this point (strip_control_chars).
# Apostrophes, hyphens, dots, commas, parentheses, ampersands and slashes stay
# allowed because real names and clinic names contain them; anything that goes
# on to build a filesystem path is separately run through
# sanitize_filename_component().
# Built from an explicit character list and re.escape()d, rather than written
# as a regex literal — hand-escaping a class containing both a backslash and
# a pipe is easy to get subtly wrong, and a silently-not-matched character
# is a hole that looks like a rule.
_NAME_FORBIDDEN_CHARS = "<>{}[]|`$;" + chr(34) + chr(92)
_NAME_FORBIDDEN = re.compile("[" + re.escape(_NAME_FORBIDDEN_CHARS) + "]")


def _name_chars_ok(text: str) -> bool:
    """True when `text` carries no character that is unsafe at a name's sinks."""
    return _NAME_FORBIDDEN.search(text) is None

# Intentionally simpler than RFC 5322: one @, a dot-bearing domain, no spaces,
# no control characters. A stricter regex rejects valid addresses; a looser one
# lets header-injection payloads through.
# Device serials come from firmware as free-form ASCII — `DM ECG V1.0 A998` is a
# real one. Spaces and dots must be allowed; path separators, quotes and shell
# metacharacters must not.
_SERIAL_ALLOWED = re.compile(r"^[A-Za-z0-9 ._\-]+$")

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

# Explicitly [^0-9], NOT \D. Python's \d is Unicode-aware, so a phone
# typed in Arabic-Indic digits (٩٨٧٦٥٤٣٢١٠) survives the strip, passes a
# ten-character length check, and is then sent to the cloud API as non-ASCII
# text. str.isdigit() has the same blind spot and is avoided below for the
# same reason.
_DIGITS_RE = re.compile(r"[^0-9]")
_ASCII_DIGITS_ONLY = re.compile(r"^[0-9]+$")


def strip_control_chars(value: Any) -> str:
    """Remove control characters, keeping ordinary printable text.

    Newlines, carriage returns, NULs and the Unicode C0/C1 categories are how a
    single form field turns into two log lines, two CSV rows, or two header
    values. They are stripped here rather than escaped at each sink, because
    there is no realistic case for a control character in a name or a phone
    number.
    """
    text = "" if value is None else str(value)
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")


def digits_only(value: Any) -> str:
    """Every digit in `value`, in order, with everything else discarded."""
    return _DIGITS_RE.sub("", "" if value is None else str(value))


def normalize_phone(value: Any) -> str:
    """Reduce a typed phone number to its ten national digits.

    Tolerates the shapes people actually paste — `+91 98765 43210`,
    `098765-43210`, `0091...` — by stripping to digits and then removing a
    leading country code or trunk zero. Returns whatever is left; the caller
    decides whether that is acceptable via `validate_phone()`.
    """
    digits = digits_only(value)
    if len(digits) > PHONE_DIGITS and digits.startswith("00"):
        digits = digits[2:]
    if len(digits) == PHONE_DIGITS + 2 and digits.startswith("91"):
        digits = digits[2:]
    if len(digits) == PHONE_DIGITS + 1 and digits.startswith("0"):
        digits = digits[1:]
    return digits


def sanitize_for_log(value: Any, max_length: int = 64) -> str:
    """Make a user-supplied value safe to put in a log line.

    Strips control characters (so one field cannot forge a second log entry) and
    truncates, so an oversized input cannot flood the log file.
    """
    text = strip_control_chars(value)
    if len(text) > max_length:
        text = text[:max_length] + "…"
    return text


def sanitize_filename_component(value: Any, fallback: str = "Unknown",
                                max_length: int = 64) -> str:
    """Reduce a user-supplied value to something safe to use as ONE path segment.

    Replacing spaces is not enough. A patient name of `../../../Users/Public/x`
    joined onto an output directory walks straight out of it — os.path.join does
    not stop at a `..`, and normpath then resolves it. Everything outside
    [A-Za-z0-9_-] is replaced, which removes dots, slashes and backslashes in one
    step, so no traversal sequence can survive.

    Also guards the Windows device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9),
    which cannot be used as file names at all.
    """
    text = strip_control_chars(value)
    cleaned = re.sub(r"[^A-Za-z0-9_\-]", "_", text)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        return fallback
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip("_") or fallback
    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved |= {f"COM{i}" for i in range(1, 10)}
    reserved |= {f"LPT{i}" for i in range(1, 10)}
    if cleaned.upper() in reserved:
        cleaned = cleaned + "_"
    return cleaned


# ══════════════════════════════════════════════════════════════════════════════
# FIELD VALIDATORS
# ══════════════════════════════════════════════════════════════════════════════
# Each returns (ok, cleaned_value, error_message). `cleaned_value` is what
# should be stored; `error_message` is empty when ok is True and is written for
# a person to read, since it goes straight into a QMessageBox.


def validate_phone(value: Any, field_label: str = "Phone number") -> Tuple[bool, str, str]:
    """Exactly ten digits, no letters.

    Rejects a value that carried non-digits rather than silently normalising it:
    someone who typed `98765abcde` meant something, and quietly turning that
    into a 5-digit number would look to them like the app lost their input.
    """
    raw = strip_control_chars(value).strip()
    if not raw:
        return False, "", f"{field_label} is required."

    normalized = normalize_phone(raw)

    # Anything left over after removing the digits and the accepted separators
    # means the user typed something that is not a phone number.
    leftovers = re.sub(r"[0-9\s\-()+]", "", raw)
    if leftovers:
        return False, "", f"{field_label} must contain digits only (no letters or symbols)."

    if len(normalized) != PHONE_DIGITS:
        return False, "", f"{field_label} must be exactly {PHONE_DIGITS} digits."

    return True, normalized, ""


def validate_password(value: Any, field_label: str = "Password") -> Tuple[bool, str, str]:
    """Length-bounded, and free of control characters.

    The maximum matters as much as the minimum: PBKDF2 hashes whatever it is
    given, so an unbounded password field is an unbounded amount of work for
    anyone who can reach the form.
    """
    text = "" if value is None else str(value)
    if not text:
        return False, "", f"{field_label} is required."
    if text != strip_control_chars(text):
        return False, "", f"{field_label} must not contain control characters."
    if len(text) < PASSWORD_MIN_LENGTH:
        return False, "", f"{field_label} must be at least {PASSWORD_MIN_LENGTH} characters."
    if len(text) > PASSWORD_MAX_LENGTH:
        return False, "", f"{field_label} must be at most {PASSWORD_MAX_LENGTH} characters."
    return True, text, ""


def validate_email(value: Any, field_label: str = "Email",
                   required: bool = True) -> Tuple[bool, str, str]:
    """A single, plausible address — bounded, one @, no whitespace."""
    text = strip_control_chars(value).strip()
    if not text:
        if required:
            return False, "", f"{field_label} is required."
        return True, "", ""
    if len(text) > EMAIL_MAX_LENGTH:
        return False, "", f"{field_label} must be at most {EMAIL_MAX_LENGTH} characters."
    if not _EMAIL_RE.match(text):
        return False, "", f"Enter a valid {field_label.lower()} address."
    return True, text, ""


def validate_name(value: Any, field_label: str = "Name",
                  min_length: int = NAME_MIN_LENGTH,
                  max_length: int = NAME_MAX_LENGTH,
                  required: bool = True) -> Tuple[bool, str, str]:
    """A person or organisation name: bounded, printable, no metacharacters."""
    text = strip_control_chars(value).strip()
    text = " ".join(text.split())          # collapse runs of whitespace
    if not text:
        if required:
            return False, "", f"{field_label} is required."
        return True, "", ""
    if len(text) < min_length:
        return False, "", f"{field_label} must be at least {min_length} characters."
    if len(text) > max_length:
        return False, "", f"{field_label} must be at most {max_length} characters."
    if not _name_chars_ok(text):
        return False, "", f"{field_label} contains characters that are not allowed."
    return True, text, ""


def validate_text(value: Any, field_label: str, max_length: int,
                  min_length: int = 1, required: bool = True) -> Tuple[bool, str, str]:
    """Free text (address, notes) — bounded and stripped of control characters."""
    text = strip_control_chars(value).strip()
    if not text:
        if required:
            return False, "", f"{field_label} is required."
        return True, "", ""
    if len(text) < min_length:
        return False, "", f"{field_label} must be at least {min_length} characters."
    if len(text) > max_length:
        return False, "", f"{field_label} must be at most {max_length} characters."
    return True, text, ""


def validate_username(value: Any, field_label: str = "Username") -> Tuple[bool, str, str]:
    """A login identifier: either a ten-digit phone or a bounded name."""
    text = strip_control_chars(value).strip()
    if not text:
        return False, "", f"{field_label} is required."
    if _ASCII_DIGITS_ONLY.match(text):
        return validate_phone(text, field_label)
    # A string that is ALL digits but not ASCII digits is a phone number to the
    # person reading it and something else entirely to the backend — ٩٨٧٦٥٤٣٢١٠
    # and 9876543210 are different identifiers that look identical on screen.
    # Names in any script are fine (see _NAME_FORBIDDEN); only this
    # digit-lookalike case is refused.
    if text.isdigit():
        return False, "", f"{field_label} must use ordinary digits 0-9."
    if len(text) < USERNAME_MIN_LENGTH:
        return False, "", f"{field_label} must be at least {USERNAME_MIN_LENGTH} characters."
    if len(text) > USERNAME_MAX_LENGTH:
        return False, "", f"{field_label} must be at most {USERNAME_MAX_LENGTH} characters."
    # A non-numeric login identifier is a person's name, so it gets the same
    # character rule. Without this the branch accepted any bounded string —
    # including one made of non-ASCII digits, which reads as a phone number to
    # a person but is not one to the backend.
    if not _name_chars_ok(text):
        return False, "", f"{field_label} contains characters that are not allowed."
    return True, text, ""


def validate_age(value: Any, field_label: str = "Age",
                 required: bool = True) -> Tuple[bool, str, str]:
    """A whole number of years inside a human range."""
    text = strip_control_chars(value).strip()
    if not text:
        if required:
            return False, "", f"{field_label} is required."
        return True, "", ""
    if not _ASCII_DIGITS_ONLY.match(text):
        return False, "", f"{field_label} must be a number."
    age = int(text)
    if age < AGE_MIN or age > AGE_MAX:
        return False, "", f"{field_label} must be between {AGE_MIN} and {AGE_MAX}."
    return True, str(age), ""


# The signup form parks this sentence in the read-only Serial ID box when no
# device answers the scan. It is UI copy, never a serial, and must not be stored
# as one — the form handlers blank it, and the logic layer blanks it again.
DEVICE_DISCONNECTED_PLACEHOLDER = (
    "RhythmUltra device connection lost. Please reconnect the device"
)


def is_device_placeholder(value: Any) -> bool:
    """True when the Serial ID box is showing the not-connected message."""
    return strip_control_chars(value).strip() == DEVICE_DISCONNECTED_PLACEHOLDER


def validate_serial(value: Any, field_label: str = "Serial ID",
                    required: bool = False) -> Tuple[bool, str, str]:
    """A device serial number.

    This field is NOT typed by anyone. The signup form fills it in automatically
    the moment a RhythmUltra is connected — `on_scan_finished()` in main.py sets
    it read-only from `send_machine_serial_command()`, which returns 16 bytes of
    raw ASCII straight from the firmware. Real serials look like
    `DM ECG V1.0 A998`: spaces and dots included.

    So the character rule follows the firmware, not what looks tidy for a
    hand-typed identifier. An allow-list of letters, digits, dash and underscore
    rejected every genuine device and blocked signup outright.

    What still applies, because the value reaches users.json, cloud payloads and
    S3 keys: a length bound and no control characters. Path separators and shell
    metacharacters stay excluded; anything that goes on to build a filesystem
    path should additionally go through `sanitize_filename_component()`.
    """
    text = strip_control_chars(value).strip()
    if is_device_placeholder(text):
        text = ""          # no device answered the scan — not a serial
    if not text:
        if required:
            return False, "", f"{field_label} is required."
        return True, "", ""
    if len(text) > SERIAL_MAX_LENGTH:
        return False, "", f"{field_label} must be at most {SERIAL_MAX_LENGTH} characters."
    if not _SERIAL_ALLOWED.match(text):
        return False, "", (
            f"{field_label} contains characters that are not allowed."
        )
    return True, text, ""


# ══════════════════════════════════════════════════════════════════════════════
# WIDGET HELPERS  (PyQt5 imported lazily — see module docstring)
# ══════════════════════════════════════════════════════════════════════════════

def apply_digit_only(line_edit: Any, max_digits: int = PHONE_DIGITS) -> bool:
    """Restrict a QLineEdit to at most `max_digits` digits — no letters.

    Uses a regular-expression validator rather than QIntValidator. QIntValidator
    is bounded by a C++ int, so `QIntValidator(0, 2147483647)` silently refuses
    every keystroke of any mobile number above 2147483647 — which is most of
    them. A digit regex has no such ceiling.

    Returns True if the validator was attached, False if Qt was unavailable
    (headless test runs), so callers can stay silent either way.
    """
    # The length cap is applied first and on its own. It is the weaker of the
    # two constraints but it never fails, so a validator that cannot be built
    # (no Qt, an unexpected widget type) must not take the cap down with it —
    # which is what happened when both lived in one try block.
    capped = apply_text_limits(line_edit, max_digits)

    try:
        from PyQt5.QtCore import QRegularExpression
        from PyQt5.QtGui import QRegularExpressionValidator
    except Exception:
        return False

    try:
        # QRegularExpressionValidator anchors the pattern implicitly, so this
        # matches the whole field, not a substring of it.
        validator = QRegularExpressionValidator(
            QRegularExpression("[0-9]{0,%d}" % int(max_digits)), line_edit
        )
        line_edit.setValidator(validator)
        return True
    except Exception:
        # The cap in `capped` still stands; only the digit filter was lost.
        return False


def apply_text_limits(line_edit: Any, max_length: int) -> bool:
    """Cap a QLineEdit's length so oversized input cannot be typed or pasted."""
    try:
        line_edit.setMaxLength(int(max_length))
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN ATTEMPT LIMITER
# ══════════════════════════════════════════════════════════════════════════════

class LoginRateLimiter:
    """Counts failed sign-in attempts per identifier and locks after too many.

    In-process and non-persistent by design. It exists to stop a person sitting
    at the machine from grinding passwords through the form, which is the threat
    this application can actually see. It does not — and cannot — protect the
    users.json hash file from someone who already has the disk; that is what the
    PBKDF2 work factor in sign_in.py is for.
    """

    def __init__(self, max_attempts: int = LOGIN_MAX_ATTEMPTS,
                 lockout_seconds: int = LOGIN_LOCKOUT_SECONDS):
        self.max_attempts = int(max_attempts)
        self.lockout_seconds = int(lockout_seconds)
        self._failures = {}      # key -> consecutive failure count
        self._locked_until = {}  # key -> monotonic deadline

    @staticmethod
    def _key(identifier: Any) -> str:
        return strip_control_chars(identifier).strip().lower()

    def _now(self) -> float:
        import time
        return time.monotonic()

    def is_locked(self, identifier: Any) -> bool:
        key = self._key(identifier)
        deadline = self._locked_until.get(key)
        if deadline is None:
            return False
        if self._now() >= deadline:
            # Lock expired — clear it so the next attempt starts clean.
            self._locked_until.pop(key, None)
            self._failures.pop(key, None)
            return False
        return True

    def seconds_remaining(self, identifier: Any) -> int:
        key = self._key(identifier)
        deadline = self._locked_until.get(key)
        if deadline is None:
            return 0
        return max(0, int(round(deadline - self._now())))

    def attempts_remaining(self, identifier: Any) -> int:
        key = self._key(identifier)
        return max(0, self.max_attempts - self._failures.get(key, 0))

    def record_failure(self, identifier: Any) -> bool:
        """Count one failed attempt. Returns True if that attempt caused a lock."""
        key = self._key(identifier)
        count = self._failures.get(key, 0) + 1
        self._failures[key] = count
        if count >= self.max_attempts:
            self._locked_until[key] = self._now() + self.lockout_seconds
            return True
        return False

    def record_success(self, identifier: Any) -> None:
        """Clear the counter — a correct password ends the streak."""
        key = self._key(identifier)
        self._failures.pop(key, None)
        self._locked_until.pop(key, None)

    def reset(self, identifier: Any = None) -> None:
        if identifier is None:
            self._failures.clear()
            self._locked_until.clear()
        else:
            self.record_success(identifier)
