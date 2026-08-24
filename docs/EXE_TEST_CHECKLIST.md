# CardioX — EXE Release Verification Checklist

**Applies to:** CardioX desktop (`dist\CARDIOX\CARDIOX.exe`) and its installer (`dist\installers\Setup_CARDIOX_<version>.exe`)
**Source of truth:** `build_exe.py`, `build_setup.py`, `build_release.ps1`, `installer/CardioX.iss`, `src/main.py`
**Document owner:** Release QA
**Last revised:** 2026-08-18

---

## 1. What this document is

The Python test suite in [tests/](../tests/) proves that *functions* behave. It runs headless, with no
display, no serial hardware, and no network. It cannot prove that the **shipped executable** behaves —
PyInstaller bundling, Program Files permissions, USB enumeration, licence heartbeats and PDF rendering
all live outside its reach.

This document is the missing half: a **unit-test-shaped acceptance suite for the built EXE**. Every
check has a stable ID, an explicit precondition, ordered steps, one unambiguous expected result, and a
tick box. A check is either ✅ **PASS**, ❌ **FAIL**, or ⏭️ **N/A** — there is no "mostly worked".

Run it in order. Later sections assume earlier sections passed.

### 1.1 Notation

| Mark | Meaning |
|---|---|
| `- [ ]` | Not yet executed |
| `- [x]` | Executed and **passed** |
| `- [!]` | Executed and **failed** — raise a defect, record the ID in §20 |
| `- [~]` | Not applicable to this build/environment — justify in §20 |

Check ID prefixes: `AUT` automated, `BLD` build, `INS` installer, `LNC` launch, `LIC` licensing,
`DEV` device, `ECG`/`HRV`/`HYP` clinical modules, `RPT` reports, `HIS` history, `CLD` cloud,
`DRV` doctor review, `DSH` dashboard, `PRF` performance, `SEC` security, `UPD` update, `UNI` uninstall,
`REG` regression.

### 1.2 Exit criteria

A build may ship only when:

1. All `AUT`, `BLD`, `INS`, `LNC`, `LIC`, `SEC` checks pass — **no exceptions, these are blocking**.
2. All `ECG`, `HRV`, `HYP`, `RPT` checks pass on at least one machine from each row of the matrix in §3.
3. Any failing `CLD`, `DRV`, `PRF`, `UPD` check has a logged defect with an accepted risk sign-off in §21.

---

## 2. Test environment preparation

Before starting, capture the environment in the table below. A checklist result is meaningless without it.

| Field | Value |
|---|---|
| Build version (`src/version.py` → `APP_VERSION`) | |
| Update channel | |
| Installer file + SHA-256 | |
| Build machine OS / Python version | |
| Test machine OS / build | |
| RAM / CPU / logical threads | |
| RhythmUltra serial under test | |
| Licence server URL baked into `.env` | |
| Tester name | |
| Date | |

**Reset the machine to a clean state before `INS-01`:**

- [ ] `PREP-01` Uninstall any existing CardioX (Settings → Apps, or `{app}\unins000.exe`).
- [ ] `PREP-02` Delete the runtime workspace `%LOCALAPPDATA%\Deckmount\CardioX` — this is where
      `runtime_dir()` in [src/utils/app_paths.py](../src/utils/app_paths.py#L25) stores the licence token,
      `users.json`, `ecg_settings.json`, reports, logs and the offline queue. Leaving it behind hides
      first-run bugs.
- [ ] `PREP-03` Confirm `%ProgramFiles%\CardioX` is gone.
- [ ] `PREP-04` Unplug the RhythmUltra device (device-absent paths are tested first).

> ⚠️ `PREP-02` deletes local patient history. Only do this on a QA machine, never on a customer install.

---

## 3. Environment matrix

Every clinical module (`ECG`, `HRV`, `HYP`) and the report suite (`RPT`) must be run on at least one
machine per row.

| Row | Spec | Why it matters |
|---|---|---|
| A — Minimum | ≤ 8 GB RAM, i3 / ≤ 4 logical threads | Triggers `is_low_spec_mode()`; antialiasing is disabled and the capture ring-buffer path is the one that was optimised (74.8 ms → 1.9 ms per second of capture). |
| B — Typical | 16 GB RAM, i5/i7, integrated GPU | The mainstream clinic machine. |
| C — Fresh Windows | No Python, no MSVC redistributables, no dev tools | Proves PyInstaller bundled every runtime dependency. |

---

## 4. Layer 0 — Automated unit tests (`AUT`)

These run before any EXE is produced. **A red suite stops the release.**

```powershell
.\.venv\Scripts\Activate.ps1
pip install pytest
python -m pytest tests/ -v
```

- [ ] `AUT-01` Full suite green.
      **Expected:** `180 passed, 7 subtests passed` and exit code `0`.
      *(Verified on 2026-08-18: 180 passed in 55.33 s.)*
- [ ] `AUT-02` `python -m pytest tests/test_cardiox_prod.py -q` → **76 passed**.
- [ ] `AUT-03` `python -m pytest tests/test_history_and_hyperkalemia.py -q` → **104 passed**.
- [ ] `AUT-04` Suite completes with **no** display server, **no** RhythmUltra attached and **no**
      network — headless is a hard requirement, because CI has none of the three.

### 4.1 What the 180 tests actually cover

`tests/test_cardiox_prod.py` — 76 tests, 13 classes:

| Class | Tests | Guards |
|---|---:|---|
| `TestAdminReportsPanelCredentials` | 6 | Admin panel accepts only env-configured credentials; no hardcoded fallback. |
| `TestLoginBypassRemoved` | 2 | Asserts no developer bypass string survives in the login flow. |
| `TestBuildMetricConclusions` | 14 | HR/PR/QRS/QTc → clinical conclusion text, incl. asystole and 3rd-degree AV block. |
| `TestNormalizeConclusions` | 5 | Placeholder removal, dedup, NSR suppressed alongside severe pathology. |
| `TestScreenLabelFallback` | 8 | Parsing `"72 BPM"` / `"160/148"` style screen labels back to numbers. |
| `TestRateQualifier` | 6 | Rate qualifiers appended correctly; hard cap of 5 conclusions. |
| `TestQtcAutoWarning` | 8 | QTc 440/460/500 thresholds; warning inserted first, never duplicated. |
| `TestFlatlineSafeguard` | 5 | Empty / flat / too-short signal and 0 BPM all classify as flatline. |
| `TestOfflineQueue` | 7 | Queue directories, disk persistence, pending stats, failed→pending retry, path healing. |
| `TestConnectivity` | 2 | Online when a socket connects; offline when all probes fail. |
| `TestGraceWindow` | 3 | Offline grace boundary arithmetic, including the exact-expiry edge. |
| `TestConclusionPipeline` | 5 | End-to-end conclusion build for AF + prolonged QTc, VT, normal, flatline. |
| `TestImportHealth` | 5 | Report generator, arrhythmia detector, offline queue, licence manager, auto-sync all import cleanly. |

`tests/test_history_and_hyperkalemia.py` — 104 tests, 11 classes:

| Class | Tests | Guards |
|---|---:|---|
| `TestFormatIndianPhone` | 13 | `+91-` prefixing only at exactly 10 digits; `None`/int/Unicode safe. |
| `TestBeatsInBoxes` | 10 | ECG paper-speed maths, zero/negative guards, proportionality. |
| `TestSafeFloat` | 11 | Coercion of `"72,5"`, `"72 BPM"`, lists, `None`, custom defaults. |
| `TestECGGridConstants` | 10 | 1 mm / 5 mm box sizes, A4 landscape dims, 50–1000 Hz sampling-rate guard. |
| `TestHistoryRowValuesCount` | 10 | History table is exactly 9 columns; "Findings" column removed. |
| `TestHistoryDateParsing` | 8 | Malformed/partial dates sort safely, newest first. |
| `TestHistorySearchFilter` | 8 | Case-insensitive, Unicode-safe, XSS-safe, list-valued findings. |
| `TestInferReportType` | 8 | Report type inferred from filename; explicit override wins. |
| `TestRiskSeverityClassifier` | 11 | VT/AF/STEMI critical, brady/PVC/QTc warning, critical beats warning. |
| `TestCircularBufferUnwrap` | 7 | Ring-buffer unwrap at all pointer positions; length invariant. |
| `TestFindingsSummaryTruncation` | 8 | 60-char cap with ellipsis; empty and `None`-bearing lists. |

**Coverage gap to be aware of:** nothing above touches PyInstaller packaging, serial I/O, PyQt widgets,
S3 transfer, or PDF pixel output. That is exactly why §5 onward exists.

---

## 5. Layer 1 — Build verification (`BLD`)

```powershell
.\build_release.ps1 -Name CARDIOX -Version 1.1.0
```

- [ ] `BLD-01` `build_release.ps1` regenerates `src/version.py` with the requested `APP_VERSION`,
      `UPDATE_CHANNEL` and `GITHUB_REPOSITORY` before compiling.
- [ ] `BLD-02` PyInstaller stage exits `0`. Any non-zero exit throws `"EXE build failed"` and must stop
      the release.
- [ ] `BLD-03` **Secret gate holds.** `_stage_env_for_distribution()` in
      [build_exe.py](../build_exe.py#L135) must not print a refusal. Deliberately prove the gate is
      live: add `TEST_SECRET_KEY=abc` to `.env`, rebuild, and confirm the build aborts with
      *"REFUSING TO BUILD — unvetted secret(s) would ship inside the installer"*. Remove the line and
      rebuild before continuing.
- [ ] `BLD-04` The staged `.env` inside `dist\CARDIOX\` contains **only** vetted keys. Diff it against
      `_DIST_ENV_KNOWN_SHIPPED` ([build_exe.py:115](../build_exe.py#L115)). Any server-side secret
      present here is a **release blocker**.
- [ ] `BLD-05` `LICENSE_SERVER_URL` in the staged `.env` points at the **production** gateway, not
      `localhost`. Overridable at build time via `BUILD_LICENSE_SERVER_URL`.
- [ ] `BLD-06` `dist\CARDIOX\CARDIOX.exe` exists, is `--windowed` (no console window on launch) and
      carries the multi-resolution icon from `assets/cardiox_logo.ico`.
- [ ] `BLD-07` File properties → Details shows the expected version string.
- [ ] `BLD-08` Bundled data present under `dist\CARDIOX\_internal\`: `assets\`, `config\` (must contain
      `clinical_config.yaml`).
      **Why:** without PyYAML + `clinical_config.yaml` the frozen app silently falls back to Python
      defaults and every arrhythmia reads *"Rhythm Undetermined"* — see the comment at
      [build_exe.py:345](../build_exe.py#L345).
- [ ] `BLD-09` No stale spec used: the spec is written to `build\`, not the repo root
      (`--specpath=build`). Confirm `build\CARDIOX.spec` has this run's timestamp.
- [ ] `BLD-10` Excluded frameworks really are absent from `_internal`: `PyQt6`, `PySide2`, `PySide6`,
      `notebook`, `IPython`, `jinja2`.
- [ ] `BLD-11` Inno Setup stage produces `dist\installers\Setup_CARDIOX_<version>.exe`.
      A missing `ISCC.exe` prints *"Setup build could not be completed"* — install Inno Setup 6 and rerun.
- [ ] `BLD-12` Record the installer SHA-256 in §2.

---

## 6. Layer 2 — Installer (`INS`)

Per [installer/CardioX.iss](../installer/CardioX.iss).

- [ ] `INS-01` Installer requires elevation (`PrivilegesRequired=admin`) and shows the UAC prompt.
- [ ] `INS-02` EULA page displays `installer/EULA.txt`; declining cancels the install cleanly.
- [ ] `INS-03` Default install directory is `{autopf}\CardioX` (i.e. `C:\Program Files\CardioX`).
- [ ] `INS-04` The whole `dist` tree is copied recursively — `{app}\_internal\` is present, not just the EXE.
- [ ] `INS-05` Start Menu group **CardioX** created with the app entry and **Uninstall CardioX**.
- [ ] `INS-06` Desktop icon created **only** when the "Create a desktop icon" task is ticked.
- [ ] `INS-07` Both shortcuts show the CardioX logo, not a generic or Python icon, and set
      `WorkingDir` to `{app}`.
- [ ] `INS-08` The post-install "Launch CardioX" checkbox starts the app successfully.
- [ ] `INS-09` Silent install `Setup_CARDIOX_<v>.exe /VERYSILENT /NORESTART` completes and does **not**
      auto-launch (`skipifsilent`).
- [ ] `INS-10` Re-running the installer over an existing install upgrades in place without orphaning
      files in `_internal`.

---

## 7. Layer 3 — First launch / cold start (`LNC`)

- [ ] `LNC-01` Double-clicking the desktop shortcut on a **clean** machine (row C) shows the CardioX
      window. No Python traceback, no missing-DLL dialog, no console flash.
- [ ] `LNC-02` **Runtime workspace is created outside Program Files.** After first launch,
      `%LOCALAPPDATA%\Deckmount\CardioX` exists and contains `reports\`, `logs\`, `offline_queue\`,
      `temp\`, `src\`.
      **Why:** `_prepare_runtime_workspace()` ([src/main.py:78](../src/main.py#L78)) must relocate all
      writes; writing under `Program Files` fails for standard users.
- [ ] `LNC-03` Seed files copied into the workspace on first run: `.env`, `customer_channels.json`,
      `users.json`, `ecg_settings.json`, `last_conclusions.json`, `src\users.json`,
      `src\ecg_settings.json`.
- [ ] `LNC-04` **Nothing is written under `C:\Program Files\CardioX`** during a full session. Verify with
      Process Monitor filtered on `Path contains CardioX` + `Operation is WriteFile`, or by checking
      that no file under `{app}` has a modified timestamp later than install time.
- [ ] `LNC-05` Run the app as a **standard (non-admin) user**. It launches and can save a report.
- [ ] `LNC-06` The Windows taskbar groups the window under the CardioX icon
      (`AppUserModelID = CardioX.1.1.0`), not under a Python interpreter icon.
- [ ] `LNC-07` A second launch while the app is running does **not** open a second instance; the
      existing window is raised (single-instance socket in `main()`).
- [ ] `LNC-08` Cold start time from double-click to a usable login screen is recorded here: `______ s`.
- [ ] `LNC-09` `%LOCALAPPDATA%\Deckmount\CardioX\logs\` receives a log file with no `ERROR`/`Traceback`
      entries from a clean start.

---

## 8. Layer 4 — Licensing & device authorization (`LIC`)

These map 1:1 to the eight business rules in the [README](../README.md#-licensing--device-authorization).
`run_startup_checks()` in `src/utils/license_manager.py` runs a fixed 5-step sequence:
**(1)** token file exists → **(2)** token integrity → **(3)** hardware fingerprint matches →
**(4)** RhythmUltra connected and serial matches → **(5)** server heartbeat.

- [ ] `LIC-01` **Rule 1 — Signup.** A fresh install with no token shows Sign Up. Completing it creates
      the account server-side and shows the generated credentials **once**.
- [ ] `LIC-02` `cardiox.lic` is written into the runtime workspace and carries the RhythmUltra serial
      that was connected at signup.
- [ ] `LIC-03` **No double registration.** Sign up once, then log in and out three times. The server must
      still show **one** active seat for this machine. *(Regression: the old build called `/register`
      twice, which deactivated the first seat and made every later heartbeat answer `SEAT_INACTIVE`.)*
- [ ] `LIC-04` **Rule 2 — 5 seats per device.** Register the same RhythmUltra serial from five different
      machines (or fingerprints); all five succeed. The **sixth** is refused with a device-limit dialog,
      not a generic error.
- [ ] `LIC-05` **Rule 3 — Revocation stops the software.** Revoke the seat server-side, restart the app.
      Expected: login blocked, message naming revocation and pointing at Deckmount support, local
      credentials wiped, user returned to Sign Up.
- [ ] `LIC-06` Both server codes — `LICENSE_REVOKED` and `ACCOUNT_REVOKED` — produce the **same**
      user-facing wording.
- [ ] `LIC-07` **Rule 4 — Revoked user cannot re-register.**
      ⚠️ **Known gap:** not enforced in `cardiox-license-register`. Record the observed behaviour and
      confirm the defect is still tracked; do not mark PASS on the current backend.
- [ ] `LIC-08` **Rule 5 — Seat release re-opens registration.** Run **both** SQL statements from the
      README (update `license_seats` *and* flip `licenses.status` back to `unused`). The user can then
      register again. Running only the first statement must be shown to strand the key — verify this,
      because it is the mistake the runbook exists to prevent.
- [ ] `LIC-09` **Rule 6 — 7-day offline grace.** Disconnect the network. The app keeps working for 7 days
      (adjust the system clock or the stored timestamp to test). On day 8 it stops.
- [ ] `LIC-10` **Rule 7 — 3-day warning.** On each launch during the final 3 days, a warning states that
      internet is required for verification or the software will stop.
- [ ] `LIC-11` **Rule 8 — Grace expiry never wipes credentials.** After grace expires, `users.json` still
      contains every local record. Reconnecting the network restores access with no re-registration.
      **This is the hard line:** `users.json` is shared clinical data.
- [ ] `LIC-12` **Gate fails closed.** Force an exception inside the licence gate (e.g. point
      `LICENSE_SERVER_URL` at an unroutable address with no cached grace). The app must **deny** access,
      never fall through to the dashboard.
- [ ] `LIC-13` **Stale-seat self-repair is silent.** Corrupt/expire the token, then log in with valid
      credentials. `resync_token_from_credentials()` refreshes from `/validate` and the checks re-run
      with **no dialog** and **no** "device seat was not found" prompt offering to wipe the licence.
- [ ] `LIC-14` `SEAT_INACTIVE` is **not** treated as revocation — the local account survives it.
- [ ] `LIC-15` **Check-5 heartbeat is unconditional.** With a valid local token present and the network
      up, confirm the heartbeat still fires. Making it conditional on local state silently disables
      revocation enforcement entirely.
- [ ] `LIC-16` **Dev auto-login is off by default.** Without `CARDIOX_DEV_AUTOLOGIN` set, no credential
      is pre-filled and no bypass exists in the shipped EXE.
- [ ] `LIC-17` Signup failure dialogs are specific: device limit, revoked, and server-unreachable each
      produce a distinct message.
- [ ] `LIC-18` **Login is concurrent, not sequential.** Measure sign-in wall time. Expected ≈ **4.5 s**,
      not ≈ 8.5 s — the heartbeat and the credential check now run in parallel. Record: `______ s`.

---

## 9. Layer 5 — Device / hardware (`DEV`)

- [ ] `DEV-01` With no device attached, the dashboard reports "not connected" and does not crash.
- [ ] `DEV-02` Plugging in the RhythmUltra is detected without restarting the app; the serial and
      firmware version are displayed.
- [ ] `DEV-03` **Hardware lock.** Attach a RhythmUltra whose serial differs from the one in
      `cardiox.lic`. Acquisition must be refused in **all three** modules — 12-Lead, HRV, Hyperkalemia.
- [ ] `DEV-04` Unplugging mid-capture is handled gracefully: acquisition stops, an error is shown, no
      crash, no half-written recording.
- [ ] `DEV-05` Re-plugging after `DEV-04` restores acquisition without an app restart.
- [ ] `DEV-06` COM port enumeration works on a machine with several serial devices present
      (`pyserial` + `serial.tools.list_ports.windows` are bundled — see `BLD-08`).

---

## 10. Layer 6 — 12-Lead ECG module (`ECG`)

- [ ] `ECG-01` 12-Lead window opens from the dashboard and all 12 traces render.
- [ ] `ECG-02` Live trace scrolls smoothly, newest sample on the right, no black speckle inside the line
      (`mode='subsample'`, not `'peak'`).
- [ ] `ECG-03` Trace colour `#00FF00` at 2.0 px, matching HRV and Hyperkalemia.
- [ ] `ECG-04` Top-bar metrics (HR, PR, QRS, QTc) populate within a few seconds and are physiologically
      plausible against a known signal source.
- [ ] `ECG-05` **No 260 BPM startup spike.** Watch the first 5 seconds after starting capture. The
      startup lockout (12 beats, 3.0 s minimum) must suppress the flat-buffer transient.
- [ ] `ECG-06` **Lead-off resets metrics.** Disconnect a lead: `heart_rate`, `pr_interval`,
      `qrs_duration`, `qtc_interval` all go to `0` / `--` immediately, and the red disconnection alert
      appears.
- [ ] `ECG-07` **Resume during lead-off.** Freeze, disconnect a lead, press **Resume** — metrics reset to
      `0 BPM` / `0 ms` / `--` instantly and the alert stays active.
- [ ] `ECG-08` A report frozen during lead disconnection prints `0` for every interval metric, not stale
      values.
- [ ] `ECG-09` **No fabricated fallbacks.** With fewer than 2 R-peaks or a flat signal, the display shows
      `0`, never the old `60 BPM` / `160/148 ms PR` defaults.
- [ ] `ECG-10` **RA disconnection invalidates chest leads.** Detach RA. V1–V6 must flatline and be listed
      in the red "Leads Off" indicator — *not* show floating WCT noise.
      *(Root cause: chest leads reference Wilson's Central Terminal `(RA+LA+LL)/3`; the hardware still
      reports `connected = True` because the chest electrodes are physically attached.)*
- [ ] `ECG-11` **No lead-off false positives on clean signal.** Run 15 minutes of clean acquisition with
      all leads attached. Zero spurious lead-off verdicts, and in particular **aVR, III and aVF are never
      marked disconnected**. This is the check that matters most: a lead-off verdict substitutes a
      constant into the display, the analysis buffer **and** the recording, so a false positive prints a
      flat strip into a clinical report.
- [ ] `ECG-12` True disconnection is still detected within ~1 s for flat, dithered-flat, rail-clipped and
      runaway-noise conditions.
- [ ] `ECG-13` Freeze / Resume cycle preserves the frozen waveform exactly and resumes live data cleanly.
- [ ] `ECG-14` Settings-screen filter changes (AC 50/60 Hz, EMG, baseline) take effect on the 12-Lead
      display. *(12-Lead is the only module that follows the settings screen — see `HRV-05`/`HYP-07`.)*

---

## 11. Layer 7 — HRV module (`HRV`)

- [ ] `HRV-01` HRV window opens and the trace renders.
- [ ] `HRV-02` **Scrolling, not raster sweep.** The trace scrolls with the newest sample pinned right.
      The old CRT-style eraser bar (80 samples + 14 px black pen) must be gone — no moving blank gap.
- [ ] `HRV-03` No black speckle inside the line (`ds=1, auto=False, mode='subsample'`).
- [ ] `HRV-04` Time-domain metrics (SDNN, rMSSD, pNN50) and intervals (PR, QRS, QT, QTc) populate and are
      plausible.
- [ ] `HRV-05` Filters are **fixed** at AC 50 Hz / EMG 25 Hz / baseline 0.5 Hz on screen and in the
      report, regardless of the settings screen.
- [ ] `HRV-06` **BPM source is correct.** The displayed HR matches `calculate_ecg_metrics()` →
      `calculate_hr_rr()` (Pan-Tompkins → median RR). A ~151 BPM display against a real ~83 BPM signal
      means `HolterBPMController` has crept back in as the primary source.
- [ ] `HRV-07` The value in the PDF and in `hyper_metric.json` matches the on-screen HR exactly.
- [ ] `HRV-08` **Lead-off zeroes metrics.** Disconnect all leads: HR/RR/PR/QRS/QT/QTc reset to 0,
      smoothing buffers clear, labels read `0 BPM` / `0 ms`, and the bold red
      *"No ECG signal detected. Check patient leads."* label appears left of the status indicator.
- [ ] `HRV-09` The canvas still shows a flatline during disconnection (raw buffering is retained).
- [ ] `HRV-10` **NaN never breaks the line.** Inject or provoke a non-finite sample; the trace holds the
      last good value rather than writing NaN into the sweep buffer. With `connect='finite'`, a single
      NaN would otherwise sever the trace.
- [ ] `HRV-11` Trace colour `#00FF00` at 2.0 px.
- [ ] `HRV-12` **Shortcut sheet is accurate.** The History window shortcut sheet must not advertise
      **Ctrl + E** — that binding was removed with the email button.

---

## 12. Layer 8 — Hyperkalemia module (`HYP`)

- [ ] `HYP-01` Hyperkalemia window opens and all lanes render.
- [ ] `HYP-02` Trace is a clean thin line, not hairy. Lanes are ~900 px wide and the signal is decimated
      2× to ≈1.7 points per pixel column (was 6.7).
- [ ] `HYP-03` Amplitude is unaffected by that decimation — the signal is low-passed at 25 Hz, so an
      effective 250 Hz is 10× Nyquist and R-wave height is preserved. Compare peak-to-peak against the
      12-Lead view of the same signal.
- [ ] `HYP-04` **Baseline follower works.** Over a 10-minute run the trace stays vertically centred. A
      trace drifting off-centre means the anchor is being measured once and locked instead of using the
      slow exponential follower (α = 0.0005) plus DC removal.
- [ ] `HYP-05` **Lead-off latching.** A brief motion artifact must **not** blank a lead. A verdict latches
      only after **0.5 s** of continuous agreement and releases after 0.2 s.
- [ ] `HYP-06` Estimated K⁺ is computed and displayed; metrics zero out on full lead disconnection with
      the same red warning label as HRV.
- [ ] `HYP-07` Filters fixed at AC 50 Hz / EMG 25 Hz / baseline 0.5 Hz, independent of the settings screen.
- [ ] `HYP-08` Non-finite samples are sanitised before `setData` — the trace never breaks.
- [ ] `HYP-09` Trace colour `#00FF00` at 2.0 px.
- [ ] `HYP-10` **Capture is cheap.** Under Task Manager during capture, CPU is materially below the old
      build. The buffer path is index-addressed rings, not per-sample `np.roll` (which copied a
      10,000-element buffer 500×/s × 7 leads ≈ 40 M element moves per second).

---

## 13. Layer 9 — Reports & PDF (`RPT`)

- [ ] `RPT-01` A 12-Lead report generates and opens in the in-app preview.
- [ ] `RPT-02` The PDF lands in `%LOCALAPPDATA%\Deckmount\CardioX\reports\` with the
      `<DEVICE>_<YYYYMMDD>_<HHMMSS>` naming convention.
- [ ] `RPT-03` Patient name, age, gender, date/time, organisation block and doctor block are all correct
      and none overflow their boxes.
- [ ] `RPT-04` Grid geometry: 1 mm minor / 5 mm major, 25 mm/s, 10 mm/mV, and the 1 mV calibration pulse
      is present on every strip.
- [ ] `RPT-05` **No edge tapers.** Inspect the first and last 200 ms of every strip. Amplitude at the
      edges must match the interior. `stabilize_report_edges()` used to cross-fade the first/last
      140–180 ms to baseline, printing a beat 60 ms from the end at **14 %** of its true height.
- [ ] `RPT-06` **6×2 strips are not trimmed.** 5000 samples recorded must be 5000 samples printed. The
      two noise-triggered trims and the unconditional 3 %-per-side hard trim (up to **16 %** loss on
      clean traces) are gone, and neither end is faded.
- [ ] `RPT-07` HRV report strips are taper-free.
- [ ] `RPT-08` Hyperkalemia report strips are taper-free.
- [ ] `RPT-09` **Mains notch is always applied.** Every generated report shows the AC filter as active
      (50 Hz). The AC setting used to be read from settings in ten places across two generators, half
      defaulting to `off`, so a report could print with no mains notch at all.
- [ ] `RPT-10` Hyperkalemia report page-2 landscape layout matches the reference: patient-info X offset
      `11.85`, org contact block at X `579` / Y `546.40`, vital parameter columns and the doctor
      signature footer all aligned.
- [ ] `RPT-11` Phone numbers print as `+91-XXXXXXXXXX` only for exactly 10 digits; other lengths print
      raw rather than malformed.
- [ ] `RPT-12` Conclusions: at most 5, no placeholders or `---`, no duplicates, NSR suppressed when a
      severe pathology is listed.
- [ ] `RPT-13` A report generated at **0 BPM** prints *"No ECG data available"* rather than a false
      interpretation.
- [ ] `RPT-14` QTc warnings appear at the 440 / 460 / 500 ms thresholds, inserted first and never doubled.
- [ ] `RPT-15` Arrhythmia classification is a real diagnosis, **not** *"Rhythm Undetermined"*, on a signal
      known to contain one. A blanket "Undetermined" in the frozen build but not in dev means
      `clinical_config.yaml` or PyYAML failed to bundle — go back to `BLD-08`.
- [ ] `RPT-16` The PDF opens correctly in Adobe Reader, Edge and Chrome, and prints on A4 at 100 % scale
      with the grid measuring a true 5 mm per large box on the paper.
- [ ] `RPT-17` Generating 10 reports back to back produces 10 distinct files with no filename collisions
      and no memory growth.

---

## 14. Layer 10 — History (`HIS`)

- [ ] `HIS-01` History window lists past reports, newest first.
- [ ] `HIS-02` Table has exactly **9** columns and no horizontal scrollbar at default window size. The
      "Findings" column was deliberately removed.
- [ ] `HIS-03` Findings text is still visible inside the in-app PDF preview panel.
- [ ] `HIS-04` Search is case-insensitive and safe against Unicode and special characters.
- [ ] `HIS-05` Malformed or partial dates in `ecg_history.json` sort safely instead of throwing.
- [ ] `HIS-06` Report type is inferred correctly from the filename (`hyperkalemia`, `hrv`, `holter`,
      `analysis`, default `ecg`).
- [ ] `HIS-07` Opening a row loads the PDF in the preview panel.
- [ ] `HIS-08` The shortcuts sheet lists only bindings that actually work (see `HRV-12`).

---

## 15. Layer 11 — Cloud sync & offline queue (`CLD`)

- [ ] `CLD-01` With the network up, ECG data offloads to S3 on the background thread roughly every
      15 seconds; the UI never blocks during upload.
- [ ] `CLD-02` **Offline capture queues, never loses.** Disconnect the network, record a full session,
      generate a report. The payload appears in `offline_queue\pending\`.
- [ ] `CLD-03` Reconnecting the network drains `pending\` into `synced\` automatically, with no user
      action.
- [ ] `CLD-04` A deliberately corrupted payload lands in `offline_queue\failed\` rather than blocking the
      queue.
- [ ] `CLD-05` Retrying a failed item moves it back to `pending\` and it then syncs.
- [ ] `CLD-06` Killing the app mid-upload and restarting resumes the queue with **no** duplicate and
      **no** lost payload.
- [ ] `CLD-07` The manual **Cloud Sync** dashboard button reports success/failure honestly.
- [ ] `CLD-08` Queue statistics on the dashboard match the actual file counts on disk.

---

## 16. Layer 12 — Doctor review upload (`DRV`)

This section exists because of a real production failure: **HTTP 404 "Old Report not supported"**.

- [ ] `DRV-01` Sending a report for doctor review returns success, not 404.
- [ ] `DRV-02` **Identity comes from the filename, not from upload-time state.** Generate a report today,
      then change the system date forward two days (or wait) and upload it. The S3 key must use the
      date **encoded in the filename**, not `datetime.now()`.
- [ ] `DRV-03` Generate a report with device A connected, disconnect it, connect device B, then upload.
      The S3 key must use **device A** — the device parsed from the filename — and must never fall back
      to `0000`.
- [ ] `DRV-04` `send_for_doctor_review()` sends a `deviceId` that matches the upload prefix exactly. The
      upload path and the assignment cannot be allowed to disagree.
- [ ] `DRV-05` **Retries do not multiply.** Force three retries of one report. Exactly **one** object
      exists in S3 afterwards. *(The old build filed one report six times under the wrong prefix.)*
- [ ] `DRV-06` All three accepted filename formats resolve correctly through `_report_identity()` /
      `_report_location()`.
- [ ] `DRV-07` The uploaded report is retrievable by the doctor-review backend, which matches on device
      id + timestamp parsed from the PDF filename and requires a matching database row.

---

## 17. Layer 13 — Dashboard, settings & auxiliary UI (`DSH`)

- [ ] `DSH-01` Dashboard shows: **12-Lead ECG**, **HRV Test**, **Hyperkalemia Test**, **History**,
      **Waveform Analysis**, **AI Chatbot**.
- [ ] `DSH-02` **Comprehensive ECG (`holter_btn`) is hidden** — the dashboard is intentionally focused on
      the core workflows.
- [ ] `DSH-03` **Sticky metric cache invalidates.** Reconnect the device or drop to 0 BPM; the dashboard
      must not keep showing a stale PR (e.g. `151 ms`) from the previous session.
- [ ] `DSH-04` At **0 BPM** the interpretation box reads *"Waiting for stable ECG data..."* — never
      *"Normal Sinus Rhythm"* and never a PR status.
- [ ] `DSH-05` Medical Mode and Dark Mode toggle correctly and survive an app restart.
- [ ] `DSH-06` Doctor Profile saves and reloads; the saved name appears on generated reports.
- [ ] `DSH-07` Patient registration accepts a new patient and that patient appears in a subsequent report.
- [ ] `DSH-08` Version Information shows the `APP_VERSION` from `BLD-01`.
- [ ] `DSH-09` Help and Support opens without error.
- [ ] `DSH-10` Sign Out returns to login and clears the session; the next login is a full authentication.
- [ ] `DSH-11` Settings changes persist to `ecg_settings.json` in the **runtime workspace**, not the
      install directory.
- [ ] `DSH-12` **Demo mode.** With no device attached, a static `.ecgh` dataset replays and drives trace
      rendering, clinical algorithms and PDF generation.
- [ ] `DSH-13` No startup `AttributeError` from `datetime` shadowing — a clean launch shows no
      `type object 'datetime.datetime' has no attribute 'datetime'` in the log.

---

## 18. Layer 14 — Performance on minimum spec (`PRF`)

Run on environment row **A** (≤ 8 GB RAM, ≤ 4 threads).

- [ ] `PRF-01` `is_low_spec_mode()` engages and antialiasing is disabled (it roughly triples line-drawing
      cost).
- [ ] `PRF-02` A 10-minute continuous capture holds a steady frame rate with no progressive slowdown.
- [ ] `PRF-03` Memory is stable across that 10 minutes — record start and end RSS: `_____ MB → _____ MB`.
- [ ] `PRF-04` CPU during capture stays within budget; the ring-buffer optimisation should return roughly
      18 % of a core versus the old `np.roll` path. Record: `_____ %`.
- [ ] `PRF-05` Report generation completes in a reasonable time and the UI recovers fully afterwards.
- [ ] `PRF-06` Login completes in ≈ 4.5 s (see `LIC-18`).
- [ ] `PRF-07` Opening and closing each module five times leaks neither memory nor window handles.

---

## 19. Layer 15 — Security (`SEC`)

- [ ] `SEC-01` **No login bypass in the shipped EXE.** No hardcoded credential, no developer account, no
      `CARDIOX_DEV_AUTOLOGIN` default-on. *(`AUT` covers the source; this covers the binary.)*
- [ ] `SEC-02` Admin panel accepts **only** the env-configured `ADMIN_PASSWORD` / `CARDIOX_ADMIN_PASS`.
      There is no fallback that works when the variable is unset.
- [ ] `SEC-03` Strings dumped from the EXE and `_internal` contain **no** server-side secret. Cross-check
      against `_DIST_ENV_KNOWN_SHIPPED`.
      ⚠️ **Known exposure, tracked, not a new defect:** `AWS_*`, the report/chatbot gateway API keys, and
      `CARDIOX_ADMIN_*` do ship today. Remediation is presigned URLs and per-install tokens. Confirm the
      set has not **grown** since the last release.
- [ ] `SEC-04` `ecg_history.json` is gitignored and no patient data is committed.
- [ ] `SEC-05` The licence token is not trivially editable — tampering with `cardiox.lic` is caught by
      Check-2 (integrity).
      ⚠️ **Known limitation:** tokens are signed with `JWT_SECRET` but verified against
      `LICENSE_HMAC_SECRET`, so offline signature verification cannot succeed. Real offline tamper
      detection needs RS256 with only the public key shipped. Verify the behaviour and confirm the item
      is still tracked.
- [ ] `SEC-06` Division-by-zero and locale guards hold: `speed_mm_per_s <= 0` is rejected, and
      `"72,5"` / `"72 BPM"` parse correctly in the hyperkalemia generator.
- [ ] `SEC-07` The app does not require admin rights to **run** (only to install).

---

## 20. Layer 16 — Update & uninstall (`UPD` / `UNI`)

- [ ] `UPD-01` The update checker runs at startup without blocking the UI.
- [ ] `UPD-02` When an update is available, the in-app banner appears, is dismissible, and repositions
      correctly on window resize.
- [ ] `UPD-03` No update available → no banner, no error dialog, no log noise.
- [ ] `UPD-04` The update check fails silently and harmlessly when offline.
- [ ] `UNI-01` Uninstalling via Start Menu or Settings → Apps removes `{app}` completely, including
      `_internal`.
- [ ] `UNI-02` Desktop and Start Menu shortcuts are removed.
- [ ] `UNI-03` **The runtime workspace survives uninstall.** `%LOCALAPPDATA%\Deckmount\CardioX` — reports,
      history, `users.json` — must **not** be deleted. That is clinical data.
- [ ] `UNI-04` Reinstalling after uninstall picks the existing workspace back up: history and reports are
      still there.

---

## 21. Regression checks for the 2026-08-13 fix set (`REG`)

Fast confirmation that this release's headline fixes are actually in the binary. Each maps to a check above.

| ID | Regression | Verified by |
|---|---|---|
| - [ ] `REG-01` | Login loop after signup | `LIC-03`, `LIC-13` |
| - [ ] `REG-02` | Error codes read from `error`, not `error_code` | `LIC-17` |
| - [ ] `REG-03` | Doctor-review 404 "Old Report not supported" | `DRV-02`, `DRV-03` |
| - [ ] `REG-04` | Duplicate uploads under the wrong prefix | `DRV-05` |
| - [ ] `REG-05` | Lead-off false positives on clean signal (aVR/III/aVF) | `ECG-11` |
| - [ ] `REG-06` | `smooth_display.py` private threshold copy removed | `ECG-11` |
| - [ ] `REG-07` | Report edge tapers | `RPT-05` |
| - [ ] `REG-08` | 6×2 strip trimming | `RPT-06` |
| - [ ] `REG-09` | HRV raster sweep eraser bar | `HRV-02` |
| - [ ] `REG-10` | `mode='peak'` speckle | `HRV-03`, `HYP-02` |
| - [ ] `REG-11` | Hyperkalemia baseline drift | `HYP-04` |
| - [ ] `REG-12` | AC filter defaulting to `off` in reports | `RPT-09` |
| - [ ] `REG-13` | `np.roll` capture cost | `HYP-10`, `PRF-04` |
| - [ ] `REG-14` | Sequential login checks | `LIC-18`, `PRF-06` |
| - [ ] `REG-15` | Dead Ctrl+E shortcut in the sheet | `HRV-12` |

---

## 22. Defect log

| # | Check ID | Severity | Summary | Steps to reproduce | Status |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |

**Severity:** *Blocker* (ship-stopping), *Major* (workaround exists), *Minor* (cosmetic).

---

## 23. Release sign-off

| Layer | Checks | Passed | Failed | N/A | Signed |
|---|---:|---:|---:|---:|---|
| AUT — Automated | 4 | | | | |
| BLD — Build | 12 | | | | |
| INS — Installer | 10 | | | | |
| LNC — Launch | 9 | | | | |
| LIC — Licensing | 18 | | | | |
| DEV — Device | 6 | | | | |
| ECG — 12-Lead | 14 | | | | |
| HRV — HRV | 12 | | | | |
| HYP — Hyperkalemia | 10 | | | | |
| RPT — Reports | 17 | | | | |
| HIS — History | 8 | | | | |
| CLD — Cloud | 8 | | | | |
| DRV — Doctor review | 7 | | | | |
| DSH — Dashboard | 13 | | | | |
| PRF — Performance | 7 | | | | |
| SEC — Security | 7 | | | | |
| UPD/UNI — Update & uninstall | 8 | | | | |
| REG — Regressions | 15 | | | | |
| **Total** | **185** | | | | |

**Release decision**

- [ ] ✅ **APPROVED** — all blocking layers green, remaining defects accepted below.
- [ ] ❌ **REJECTED** — blocking failures recorded in §22.

| Role | Name | Signature | Date |
|---|---|---|---|
| QA Engineer | | | |
| Engineering Lead | | | |
| Clinical Reviewer | | | |
