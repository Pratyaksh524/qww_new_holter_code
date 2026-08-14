# 🫀 CardioX - Professional ECG & Cardiac Analysis Platform

CardioX is a clinical-grade, hardware-integrated desktop application designed for real-time ECG monitoring and advanced cardiac analysis. The platform processes, classifies, and tracks cardiac metrics with professional medical-grade precision.

---

## 🌟 Key Features

### 1. Rebranded Premium Medical Identity (CardioX)
- **Harmonious Theme:** Tailored UI utilizing orange, tech blue, and medical teal colors.
- **Custom Visual Assets:** High-resolution transparent logo (`assets/cardiox_logo.png`) and fully compliant multi-resolution Windows icon (`assets/cardiox_logo.ico`).
- **Windows Taskbar Integration:** Declares an explicit Windows `AppUserModelID` (`CardioX.1.1.0`) so the OS correctly clusters the custom CardioX taskbar icon instead of fallback Python interpreter icons.
- **Clean Dashboard Layout:** Hides the "Comprehensive ECG" button (`self.holter_btn`) to keep the dashboard focused entirely on core workflows.

### 2. Dedicated Hardware Lock (Device Authorization)
- **Secure Signup Association:** During license signup/registration, the serial number of the user's specific RhythmUltra hardware is bound and stored inside the license token (`cardiox.lic`).
- **Acquisition Lockdown:** Real-time data acquisition in the three core modules (**12-Lead ECG**, **HRV**, and **Hyperkalemia**) is restricted. The software strictly validates the connected physical USB device against the signup token serial number, preventing unauthorized hardware usage.

### 3. Professional Medical Dashboard & Diagnostics
- **High-Fidelity Trace Viewer:** Sub-pixel waveform rendering using PyQtGraph with strict grid snapping and dynamic overlay pins.
- **Beat Morphology Classification:** Real-time event tagging (N, V, S, AF) and clustering of abnormal beats (VE, SVE) for rapid medical review.
- **HRV Analytics:** Real-time calculations of Time Domain metrics (SDNN, rMSSD, pNN50) and interval stats (PR, QRS, QT, QTc).

### 4. Zero-Collision Cloud Synchronization
- **Asynchronous S3 Offloading:** Automatic background thread offloading of ECG data to AWS S3 every 15 seconds.
- **Local Cache Queueing:** Zero data loss offline-first architecture; queued files auto-sync the moment internet access is restored.

---

## 🛠️ Tech Stack
- **Core Engine:** Python 3.10
- **User Interface:** PyQt5, PyQtGraph (Hardware-accelerated rendering)
- **Signal Analysis:** NumPy, SciPy (FFT, digital filtering)
- **Report Generation:** ReportLab, Matplotlib (Agg background renderer)
- **Packaging & Setup:** PyInstaller, Inno Setup 6

---

## 🚀 Quick Start (Developers)

```powershell
# 1. Activate your virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the application
python src\main.py
```

---

## 📦 Release Compilation & Installer Packaging

CardioX uses a unified release pipeline script to compile the application and generate a single-file Windows setup installer:

```powershell
# Run the release script to compile and build the installer
.\build_release.ps1 -Name CARDIOX -Version 1.1.0
```

### Build Details:
1. **PyInstaller Compilation:** Packages python scripts into a standalone executable. Uses `--icon=assets/cardiox_logo.ico` to embed the square padded multi-resolution logo directly into the output binary (`dist\CARDIOX\CARDIOX.exe`).
2. **Inno Setup Integration:** Invokes Inno Setup compiler (`ISCC.exe`) using [CardioX.iss](file:///c:/Users/DELL/Documents/QW/qww/installer/CardioX.iss) to package the setup program, apply shortcut icon mappings (Desktop, Start Menu, Uninstaller), and create the final installer.
3. **Installer Output:** Generates the ready-to-run installation executable at `dist\installers\Setup_CARDIOX_1.1.0.exe`.

---

## 📝 Admin & Demo Configuration

**Admin Dashboard Credentials:**
- **User:** `admin` / **Password:** `adminsd`

**Demo Mode:**
A simulated hardware mode is supported. If no device is connected, static `.ecgh` datasets can be replayed to demonstrate trace rendering, clinical algorithms, and PDF report generation.

---
**Status:** 🟢 Production Ready | **Last Updated:** August 2026

---

## 🧪 Running Tests

The project includes a headless unit test suite (no display or hardware required):

```powershell
# Install pytest if not present
pip install pytest

# Run the full test suite
python -m pytest tests/ -v

# Run only the history/hyperkalemia tests
python -m pytest tests/test_history_and_hyperkalemia.py -v

# Run only the production deployment tests
python -m pytest tests/test_cardiox_prod.py -v
```

> **Current Coverage:** 180 tests across 21 test classes covering authentication, signal processing, PDF generation, offline queue, connectivity, and clinical metric classification — **100% PASSING**.

---

## 🔐 Licensing & Device Authorization

The licensing rules below are the product specification. Treat them as authoritative
when changing `src/utils/license_manager.py`, the login flow in `src/main.py`, or the
`cardiox-license-*` Lambdas.

### Business rules

| # | Rule |
|---|---|
| 1 | **Signup** creates the account server-side and shows the user their credentials once. |
| 2 | **One RhythmUltra device allows 5 signups** — five *different* machines may share one device serial (e.g. `DM ECG V1.0 A010`). |
| 3 | **Revocation stops the software.** Login is blocked, the user is told the licence is revoked and to contact Deckmount support, every local credential is wiped, and they are returned to Sign Up. |
| 4 | **A revoked user may not sign up again.** The server must refuse re-registration. Otherwise revocation is meaningless — the customer is running again in under a minute. |
| 5 | **Releasing a seat is a separate admin action** that re-opens registration. Only after a seat is explicitly freed may that user register again. |
| 6 | **Offline grace is 7 days.** A periodic server heartbeat is mandatory — it is what prevents a one-time signup being used indefinitely on a machine that never contacts the server again. |
| 7 | **The final 3 days of the offline window show a warning** on every launch: internet is required for verification or the software will stop. |
| 8 | **Exhausting offline grace must never wipe credentials.** Unlike revocation, the user simply needs to get back online. Local `users.json` records are never deleted — that is shared clinical data. |

Revocation is detected **only** by the Check-5 heartbeat inside `run_startup_checks()`.
Never make that call conditional on local state such as a token file existing; doing so
silently disables revocation enforcement entirely.

### Enforced in the client

| Behaviour | Where |
|---|---|
| Revoked seat blocks login, wipes credentials, returns to Sign Up | `src/main.py` |
| Licence gate **fails closed** — an error in the gate denies access rather than falling through to login | `src/main.py` |
| Developer auto-login gated behind `CARDIOX_DEV_AUTOLOGIN` (default off) | `src/main.py`, `src/auth/sign_in.py` |
| 7-day offline grace with a 3-day countdown warning | `src/utils/license_manager.py` |
| Signup failures show a specific dialog — device limit, revoked, or server unreachable | `src/utils/license_dialog.py` |
| Server-side secrets stripped from the distributed `.env`; build aborts on an unvetted secret | `build_exe.py` |

Both revocation codes the server emits (`LICENSE_REVOKED` for a revoked licence,
`ACCOUNT_REVOKED` for a revoked seat) map to the same user-facing message, so the wording
does not depend on which one arrived.

### Enforced in the database

Applied by `license_server/migrations/001_seat_integrity.sql`:

- unique index on `(license_id, bound_fingerprint)` — one live seat per machine per licence
- unique index on `(license_id, seat_number)`
- `CHECK` constraint restricting `plan_type` to `single` / `clinic` / `hospital` / `enterprise`
- `trg_supersede_duplicate_seat` — **temporary**, substituting for the fingerprint dedup the register Lambda is missing

**Releasing a seat requires two statements.** Updating `license_seats` alone leaves the
licence at `status='active'` with no seats, and the register path only draws licences
whose status is `'unused'`, so the key is stranded permanently:

```sql
UPDATE license_seats
   SET status='revoked', revoked_at=NOW(), released_at=NOW()
 WHERE rhythmulta_serial = :serial AND status='active';

UPDATE licenses l SET status='unused'
 WHERE l.id = :license_id
   AND NOT EXISTS (SELECT 1 FROM license_seats s
                    WHERE s.license_id = l.id AND s.status='active');
```

Seat status values: `active` (live), `revoked` (admin block), `superseded` (retired by
the dedup trigger when the same machine re-registered — not a licensing decision).

### Known gaps

Not yet implemented in `cardiox-license-register`:

- **Rule 4 is not enforced** — a revoked user can still sign up again. Needs a refusal when
  a seat exists for the fingerprint with `status='revoked' AND released_at IS NULL`.
- `license_key` in the request body is ignored entirely; a key absent from the database
  returns the same response as a valid one.
- No hardware fingerprint dedup, which is why the database trigger exists.
- `seat_number` is derived as `COUNT(*)+1` rather than `MAX(seat_number)+1`, so it
  collides after any row deletion.
- Resolve → check → insert is not wrapped in a transaction, so two concurrent signups can
  both pass the seat-cap check.

Client-side, tokens are signed by the server with `JWT_SECRET` but verified against
`LICENSE_HMAC_SECRET`, so offline signature verification cannot succeed. Real offline
tamper detection requires moving to RS256 and shipping only the public key.

---

## 📋 Changelog

### 🔧 [2026-08-13] — Licence Re-Sync, Doctor-Review Upload, Lead-Off Detection, Report Strips & Live Display

#### ✅ Login loop after signup (`src/auth/sign_in.py`, `src/main.py`, `src/utils/license_manager.py`)
- **Double registration removed:** signup called `/register` twice — once via `license_manager.register_device()` (whose token is saved to `cardiox.lic`) and again from `register_user_with_details()`. The server moved the machine onto a new seat and deactivated the first, so every later heartbeat answered `SEAT_INACTIVE`.
- **Error code parsing:** `run_startup_checks()` read only `error_code`, but the licence Lambdas put the code in `error`. Every refusal collapsed into the `LICENSE_BLOCKED` fallback, which the login gate reported as *"Your device seat was not found"* and offered to fix by wiping the licence — creating another seat and repeating the failure on the next login.
- **Silent repair:** added `is_stale_seat_error()` and `resync_token_from_credentials()`. A stale token is refreshed from `/validate` using the credentials just entered and the checks re-run, with no dialog. Startup defers to the login screen instead of prompting.
- **Credential form:** `validate_with_credentials()` sends the SHA-256 form that `register_device()` used, retries the raw password for seats created by older builds, and falls back to the registered phone when the typed identifier resolves an old seat.
- **`SEAT_INACTIVE` is no longer treated as revocation** in `sign_in`, which previously deleted the local account and wiped the licence over an out-of-date token.

#### ✅ Doctor-review upload — HTTP 404 "Old Report not supported" (`src/utils/cloud_uploader.py`)
- S3 keys were built from upload-time state: `datetime.now()` for the date segment and the *currently connected* RhythmUltra for the device folder, falling back to `0000`. A report recorded on the 10th but uploaded on the 12th was filed where the review Lambda never looks, and each retry repeated the mistake — one report appeared six times under the wrong prefix.
- `_report_identity()` / `_report_location()` now derive both the device and the date from the report's own filename (`<DEVICE>_<YYYYMMDD>_<HHMMSS>`), covering all three filename formats the backend accepts, and `send_for_doctor_review()` takes its `deviceId` from the same resolver so the upload and the assignment cannot disagree.

#### ✅ Lead-off false positives (`src/ecg/lead_off_detection.py`, `src/ecg/smooth_display.py`, `src/ecg/hyperkalemia_test.py`)
- Thresholds fired on healthy signal. Across 857 clean one-second windows from this hardware, leads reach 3444 ADC p-p and a variance of 360,460 — above the old `amplitude_max` 3000 and `variance_max` 250,000 — and the absolute `min <= 10` rule marked the derived leads (III, aVR, aVF) disconnected for entire recordings, aVR permanently. Because a lead-off verdict substitutes a constant in the display, the analysis buffer **and** the recording, this printed flat strips into reports.
- Rewritten around three real signatures: flatline, sustained pinning at the signal's own extreme, and runaway variance. **0 false positives** on all 857 clean windows, with flat, dithered-flat, rail-clipped and runaway-noise signals still detected.
- `smooth_display.py` carried a private copy of the same broken thresholds; it now delegates to the shared detector.
- The hyperkalemia window latches a verdict only after **0.5 s** of continuous agreement and releases after 0.2 s, so a motion artifact can no longer blank a lead.

#### ✅ Report waveform strips (`src/ecg/ecg_report_generator.py`, `src/ecg/6_2_ecg_report_generator.py`, `src/ecg/ecg_report_android.py`)
- **Edge tapers removed.** `stabilize_report_edges()` cross-faded the first and last 140–180 ms into a flat baseline; a beat 60 ms from the strip end printed at **14 %** of its true height. Filter transients are already prevented upstream by 0.5 s of real pre-roll plus reflect padding.
- **6×2 strips were also being cut.** Two noise-triggered trims and an unconditional 3 %-per-side "hard trim" discarded up to **16 %** of the recorded strip even on clean traces, and the drawing code faded both ends. All removed — 5000 samples in now means 5000 out.
- HRV and hyperkalemia report strips were already free of tapers and needed no change.

#### ✅ Live display (`src/ecg/hyperkalemia_test.py`, `src/ecg/hrv_test.py`)
- **Scrolling instead of raster sweep.** The HRV window's CRT-style eraser bar (80 samples plus a 14 px black pen) blanked part of the trace as it swept; the trace now scrolls with the newest sample pinned to the right edge.
- **Downsampling mode.** Both windows used `auto=True, mode='peak'`, which collapses each pixel column into a min/max pair and draws the trace as vertical bars — the background showed between them as black speckle in the line. Both now use the 12-lead's `ds=1, auto=False, mode='subsample'`.
- **Point density.** Hyperkalemia lanes are ~900 px wide, so plotting 6000 points meant 6.7 per pixel column and a hairy line. Now decimated 2× to 1.7 per column, matching the 12-lead. The signal is low-passed at 25 Hz, so an effective 250 Hz is ten times Nyquist — R-wave amplitude is preserved exactly.
- **Non-finite guard.** Both curves draw with `connect='finite'`, so a single NaN breaks the line. Hyperkalemia sanitises before `setData`; HRV holds the last good value rather than writing NaN into the sweep buffer, where `np.clip` leaves it as NaN and the step guard cannot catch it.
- **Baseline follower.** The hyperkalemia display anchor was measured once and locked forever, so drift pushed the trace off centre. It now uses the 12-lead's slow exponential follower (α = 0.0005) plus DC removal.
- **Filters.** AC 50 Hz / EMG 25 Hz / baseline 0.5 Hz are fixed for HRV and hyperkalemia on screen and in their reports; only the 12-lead follows the settings screen. The AC filter previously came from settings in ten places across the two report generators, half of them defaulting to `off` — so a report could print with no mains notch at all.
- **Trace style:** `#00FF00` at 2.0 px in both windows, matching the green the holter modules already use.

#### ✅ Performance — 8 GB / i3 minimum spec (`src/ecg/hyperkalemia_test.py`, `src/main.py`, `src/utils/license_manager.py`)
- **Buffer writes:** the capture loop called `np.roll` per sample per lead, copying a 10,000-element buffer 500 times a second for each of 7 leads, plus a Lead II buffer that is never read during capture — about 40 million element moves per second. Replaced with index-addressed rings: **74.8 ms → 1.9 ms** per second of capture (40× cheaper), roughly 18 % of a core returned on a minimum-spec machine.
- **Antialiasing** is switched off under `is_low_spec_mode()` (≤8 GB RAM or ≤4 threads), where it roughly triples line-drawing cost.
- **Login wait halved:** the licence heartbeat (~3.8 s) and the credential check (~3.8 s) ran sequentially on the UI thread. They are independent, so they now run concurrently — measured **8.48 s → 4.50 s**. The gate still fails closed if the worker stalls. The credential check also stops retrying the same identifier with the other password form after a seat-state answer, which cannot change the outcome.

#### ✅ UI
- Removed the **Ctrl + E** row from the History window shortcuts sheet (`src/dashboard/history_window.py`); the binding had already been removed with the email button, so the sheet advertised a shortcut that did nothing.

### 🔧 [2026-08-03] — Security, Report Formatting, 12-Lead Freeze/Resume & Dashboard Fixes

#### ✅ Security Hardening & Loophole Audit
- **Admin Panel Authentication (`src/dashboard/admin_reports.py`):** Strictly enforced custom environment passwords (`ADMIN_PASSWORD` / `CARDIOX_ADMIN_PASS`) in `_check_admin_credentials()` without hardcoded fallback bypasses.
- **Phone Number Validation (`src/ecg/hyperkalemia_ecg_report_generator.py`):** Updated `format_indian_phone()` to require a 10-digit count before adding `+91-`. Preserves raw input for invalid digit counts to prevent malformed numbers.
- **Division-by-Zero Protection (`src/ecg/hyperkalemia_ecg_report_generator.py`):** Added `speed_mm_per_s <= 0` guard to `beats_in_boxes()`.
- **Locale & Unit Parsing (`src/ecg/hyperkalemia_ecg_report_generator.py`):** Enhanced `_safe_float()` to handle comma decimal separators (`"72,5"`) and trailing unit strings (`"72 BPM"`).
- **Patient Data Privacy (`.gitignore`):** Added `ecg_history.json` to `.gitignore` to prevent local patient history records from being committed.

#### ✅ Hyperkalemia Report Generator Formatting Sync
- **Layout Alignment (`src/ecg/hyperkalemia_ecg_report_generator.py`):** Synchronized all Page 2 landscape coordinates with commit `0e0d8e1`:
  - Patient info X-offset: `13.85` → `11.85`. Adjusted Y positions for Name, Age, Gender, Report Type, Date/Time.
  - Organization contact block: X `590` → `579`, Y `545.90` → `546.40`.
  - Vital parameters: Left column HR/PR/QRS/RR/QT Y alignments; Right column QTc/QTCF/Est. K+ X/Y alignments.
  - Doctor signature footer: Shifted reference, doctor name, and signature lines down by 5-6 points and aligned X offsets.

#### ✅ 12-Lead ECG Freeze/Resume & Lead-Off Metric Fixes
- **Lead Off Reset (`src/ecg/twelve_lead_test.py`):** Corrected `reset_metrics_to_zero()` logic to check `_lead_off_latched`, ensuring metric labels (`heart_rate`, `pr_interval`, `qrs_duration`, `qtc_interval`) reset to 0 immediately when leads disconnect.
- **Resume Button Reset (`src/ecg/twelve_lead_test.py`):** Added an explicit lead-off check in `_resume_live_view()` so clicking **Resume** while leads are disconnected instantly resets top-bar metrics to `0 BPM` / `0 ms` / `--` and keeps the lead disconnection alert active.
- **Frozen Report Sanitization (`src/ecg/twelve_lead_test.py`):** Updated `_freeze_current_view()` to force `0` for all interval metrics when frozen during lead disconnection.
- **Fallback Elimination (`src/ecg/twelve_lead_test.py`):** Replaced default `60 BPM` and `160/148 ms PR` fallback values with `0` when fewer than 2 R-peaks are detected or when signal is flat.

#### ✅ Dashboard Sticky Metric Cache & Interpretation Fixes
- **Sticky Cache Invalidation (`src/dashboard/dashboard.py`):** Fixed `_dashboard_last_valid` cache that was retaining old PR values (e.g. `151 ms`) upon device reconnect or 0 BPM. Invalidated cached metrics whenever live HR is 0.
- **0 BPM Interpretation Guard (`src/dashboard/dashboard.py`):** Added `if hr <= 0:` guard in `update_ecg_interpretation()`. The dashboard interpretation box now displays *"Waiting for stable ECG data..."* when HR is 0 BPM instead of outputting false PR status or *"Normal Sinus Rhythm"*.

#### ✅ History Table Cleanup
- **Removed "Findings" Column (`src/dashboard/history_window.py`):** Column count reduced from 10 → 9; `"Findings"` removed from table headers and value list to eliminate horizontal scrolling. Findings text remains visible inside the in-app PDF preview panel.

#### ✅ Added: `tests/test_history_and_hyperkalemia.py` — 104 new unit tests
New test file covering 11 test suites:
- `TestFormatIndianPhone` — 13 edge cases (None, empty, prefixes, integers, Unicode-safe)
- `TestBeatsInBoxes` — 10 cases (ECG paper-speed math, zero guards, proportionality)
- `TestSafeFloat` — 11 cases (type coercion, unit strings, list input, defaults)
- `TestECGGridConstants` — 10 cases (A4 landscape dims, box sizes, sampling-rate guards)
- `TestHistoryRowValuesCount` — 10 cases (9-column assertion, Findings removal, fallbacks)
- `TestHistoryDateParsing` — 8 cases (malformed dates, partial dates, sort order)
- `TestHistorySearchFilter` — 8 cases (case-insensitive, Unicode, XSS-safe, list findings)
- `TestInferReportType` — 8 cases (filename-based inference, explicit override, case)
- `TestRiskSeverityClassifier` — 11 cases (critical/warning/normal keywords, list fields)
- `TestCircularBufferUnwrap` — 7 cases (ptr boundaries, NumPy arrays, length invariants)
- `TestFindingsSummaryTruncation` — 8 cases (60-char cap, ellipsis, empty list)

---

### 🔧 [2026-07-30] — Lead Disconnection & Signal Handling Fixes

#### ❌ Bug Fixed: V1–V6 chest leads showing garbage noise when RA (Right Arm) lead is disconnected

**Root Cause:** Chest leads (V1–V6) are measured against Wilson's Central Terminal (WCT = `(RA + LA + LL) / 3`). When RA or limb leads disconnect, WCT floats and breaks. However, because physical chest electrodes remain attached, the hardware returned `connected = True`, passing chaotic floating ADC noise to V1–V6 while limb leads were flatlined.

**Fix Applied (`src/ecg/serial/packet_parser.py`):**
- When limb source leads (`I`, `II`) are disconnected (`None`), the parser now explicitly invalidates all chest leads (`V1` through `V6` set to `None`).
- Replaced garbage noise waveforms on chest leads with clean flatlines and added them to the red "Leads Off" indicator, aligning PC behavior with the Android app.

#### ❌ Bug Fixed: HRV and Hyperkalemia tests holding stale metrics during lead disconnection

**Root Cause:** Unlike the 12-lead ECG view, the HRV and Hyperkalemia test modules did not automatically force calculated interval metrics (`HR`, `RR`, `PR`, `QRS`, `QT`, `QTc`) to zero when all patient leads were disconnected.

**Fix Applied (`src/ecg/hrv_test.py`, `src/ecg/hyperkalemia_test.py`):**
- Updated `update_metrics()` in both tests to check lead connection and signal variance (< 5.0 std-dev).
- When all leads disconnect: internal metric attributes reset to 0, smoothing buffers clear, and display labels show `0 BPM` / `0 ms`.
- Added `"No ECG signal detected. Check patient leads."` warning label in bold red placed to the left of the status indicator in top header bar (matching 12-lead screen).
- Maintained raw data buffering so canvas flatline visual representation remains intact.

---

### 🔧 [2026-07-29] — BPM Accuracy & Stability Fixes

#### ❌ Bug Fixed: HRV & Hyperkalemia showing wrong BPM on display and PDF report

**Root Cause:** `HolterBPMController` (a Holter-optimized 30-second window algorithm) was being used as the **primary BPM source** for both the live display label and `hyper_metric.json` / report generation. This caused the display to show values like `151 BPM` while the terminal (using the correct ECG algorithm) showed `83–85 BPM`.

**Fix Applied (`src/ecg/hrv_test.py`, `src/ecg/hyperkalemia_test.py`):**
- `_refresh_holter_bpm_label()` is now a **no-op** for HR display — it no longer overwrites the correct ECG-derived BPM every 2 seconds.
- `update_metrics()` now uses `calculate_ecg_metrics()` → `calculate_hr_rr()` (Pan-Tompkins R-peak detection → median RR → BPM) as the **primary source** — same algorithm used by the 12-lead ECG display.
- `HolterBPMController` is retained for **arrhythmia detection only**, used as a last-resort BPM fallback when `calculate_ecg_metrics()` returns 0 (signal too short).
- `self.last_heart_rate` is now synced from `calculate_ecg_metrics()`, so `hyper_metric.json` and PDF reports match the display.

#### ❌ Bug Fixed: BPM jumps to ~260 BPM for first few seconds on capture start

**Root Cause:** The ECG data buffer is initialized with flat `2048` ADC values. When real ECG samples first roll in, the sharp flat→signal transition creates fake R-peaks with RR intervals ~230ms → **260 BPM**. The startup filter only rejected too-slow RR intervals (> 6500ms), not too-fast ones.

**Fix Applied (`src/ecg/ecg_calculations.py`, `src/ecg/hrv_test.py`, `src/ecg/hyperkalemia_test.py`):**
- `calculate_hr_rr()` startup filter now rejects **both** too-long (< 10 BPM) AND too-short (> 200 BPM) RR intervals during initialization.
- `_STARTUP_LOCKOUT_BEATS` increased from `5` → `12` to cover more of the initialization transient.
- `update_metrics()` minimum wait increased from **0.5 seconds** → **3.0 seconds** of captured data before first BPM calculation, ensuring the flat buffer region has fully rolled out.

#### ❌ Bug Fixed: `datetime` AttributeError causing application startup crash

**Root Cause:** Multiple methods in `dashboard.py` had local `from datetime import datetime` imports inside function bodies, shadowing the module-level `import datetime`. This caused `AttributeError: type object 'datetime.datetime' has no attribute 'datetime'` on startup.

**Fix Applied (`src/dashboard/dashboard.py`):**
- Removed all method-level `from datetime import datetime` imports.
- All time calls now use the top-level `import datetime` → `datetime.datetime.now()`.
- Added `hasattr(self, 'metric_labels')` guard in `animate_heartbeat()` to prevent startup race condition.

