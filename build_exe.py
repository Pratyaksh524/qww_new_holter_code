"""
Deterministic Windows build script for CardioX executable.

Goals:
Produce consistent output across machines.
Default to ONEDIR packaging (more stable than onefile for this app).
Include required runtime files/assets explicitly.

Usage:
    python build_exe.py
    python build_exe.py --onefile        # optional, less stable for this app
    python build_exe.py --console        # debug startup crashes
    python build_exe.py --name CardioX
    python build_exe.py --onefile --runtime-tmpdir "C:\\Temp\\CardioX"
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import PyInstaller.__main__


def _encode_stdio_utf8_on_windows() -> None:
    if sys.platform != "win32":
        return
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def _data_sep() -> str:
    return ";" if os.name == "nt" else ":"


def _stage_runtime_file(src: Path, staging_dir: Path, dst_name: str | None = None) -> Path | None:
    """Copy a runtime file into a staging dir so build steps never mutate the repo."""
    if not src.exists():
        return None
    staging_dir.mkdir(parents=True, exist_ok=True)
    target = staging_dir / (dst_name or src.name)
    try:
        shutil.copy2(src, target)
    except Exception:
        return None
    return target


def _stage_json_placeholder(staging_dir: Path, dst_name: str) -> Path:
    """Create a minimal JSON file in the staging dir for missing optional runtime data."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    target = staging_dir / dst_name
    if not target.exists():
        try:
            target.write_text("{}", encoding="utf-8")
        except Exception:
            pass
    return target


_DEFAULT_DIST_LICENSE_SERVER_URL = "https://m4qoae4d8e.execute-api.us-east-1.amazonaws.com/prod/api/v1"


# Secrets that must NEVER reach a customer machine. The desktop app does not read
# any of these at runtime:
#   * LICENSE_API_TOKEN is the license *admin* token — it authorises /admin/licenses,
#     /admin/create and /admin/revoke. Verified unnecessary for the client: the
#     /api/v1/* endpoints accept requests with no X-API-Key header at all.
#   * LICENSE_HMAC_SECRET signs client requests, but the server does not verify the
#     signature (its HMAC_SECRET is a different value), so omitting it changes nothing.
#   * The remaining entries belong to cloud backends that are not in use — CLOUD_SERVICE
#     is 's3', so the Azure / GCS / FTP / Dropbox credentials are dead weight.
_DIST_ENV_DENY: set[str] = {
    "LICENSE_API_TOKEN",
    "LICENSE_HMAC_SECRET",
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_CONTAINER_NAME",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GCS_BUCKET_NAME",
    "DROPBOX_ACCESS_TOKEN",
    "FTP_HOST",
    "FTP_PORT",
    "FTP_USERNAME",
    "FTP_PASSWORD",
    "FTP_REMOTE_PATH",
    "EMAIL_PASSWORD",
    "SUPPORT_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    # Developer auto-login bypass. Skips the password prompt AND the licence gate
    # (heartbeat / revocation / offline grace), so it must never reach a customer.
    "CARDIOX_DEV_AUTOLOGIN",
}

# Substrings that indicate a secret. Any *new* key matching one of these must be
# explicitly acknowledged in _DIST_ENV_KNOWN_SHIPPED below, otherwise the build
# aborts — so adding a credential to .env can never silently ship to customers.
_DIST_ENV_SECRET_HINTS: tuple[str, ...] = (
    "SECRET", "PASSWORD", "TOKEN", "_KEY", "APIKEY", "CREDENTIAL", "PRIVATE",
)

# Credentials that still ship because the app genuinely needs them today.
# Each one is a known exposure, not an endorsement — see the remediation notes:
#   * AWS_* — cloud_uploader.py talks to S3 directly (CLOUD_SERVICE=s3). Should be
#     replaced by presigned URLs issued by the backend, so no key ships at all.
#   * *_API_KEY for the report/chatbot gateways — should become per-install tokens
#     issued at registration rather than one shared key baked into every copy.
#   * CARDIOX_ADMIN_* — identical admin password on every install; should be removed
#     or made per-install.
_DIST_ENV_KNOWN_SHIPPED: set[str] = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "DOCTOR_REVIEW_API_KEY",
    "DOCTOR_UPLOAD_API_KEY",
    "REVIEWED_REPORTS_API_KEY",
    "PUBLIC_API_KEY",
    "BACKEND_API_KEY",
    "CLOUD_API_KEY",
    "ECG_UNIFIED_API_KEY",
    "CHATBOT_API_KEY",
    "GROQ_API_KEY",
    "ECG_UPDATE_TELEMETRY_API_KEY",
    "CARDIOX_ADMIN_USER",
    "CARDIOX_ADMIN_PASS",
    "ADMIN_USERNAME",
    "ADMIN_PASSWORD",
}


def _stage_env_for_distribution(src: Path, staging_dir: Path) -> Path | None:
    """
    Stage a .env file for distribution.

    Behavior:
    - Copy `.env` into staging and normalize LICENSE_SERVER_URL to the remote
      production gateway used by the distributed app.
    - Drop every entry in _DIST_ENV_DENY so server-side secrets never ship.
    - Abort the build if an unrecognised secret-looking key would be distributed.
    - You can override the distributed URL via BUILD_LICENSE_SERVER_URL at build time.
    """
    if not src.exists():
        return None

    staging_dir.mkdir(parents=True, exist_ok=True)
    target = staging_dir / src.name

    try:
        raw = src.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return _stage_runtime_file(src, staging_dir)

    dist_url = os.getenv("BUILD_LICENSE_SERVER_URL", "").strip() or _DEFAULT_DIST_LICENSE_SERVER_URL

    out_lines: list[str] = []
    shipped_names: list[str] = []
    dropped: list[str] = []

    for line in raw:
        stripped = line.strip()
        if stripped.startswith("LICENSE_SERVER_URL="):
            out_lines.append(f"LICENSE_SERVER_URL={dist_url}")
            shipped_names.append("LICENSE_SERVER_URL")
            continue

        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out_lines.append(line)
            continue

        name = stripped.split("=", 1)[0].strip()
        if name in _DIST_ENV_DENY:
            dropped.append(name)
            continue

        out_lines.append(line)
        shipped_names.append(name)

    if not any(line.strip().startswith("LICENSE_SERVER_URL=") for line in out_lines):
        out_lines.append(f"\nLICENSE_SERVER_URL={dist_url}")

    # Fail closed: a secret-looking key that nobody has vetted must not ship.
    unvetted = [
        name for name in shipped_names
        if any(hint in name.upper() for hint in _DIST_ENV_SECRET_HINTS)
        and name not in _DIST_ENV_KNOWN_SHIPPED
    ]
    if unvetted:
        raise SystemExit(
            "REFUSING TO BUILD — unvetted secret(s) would ship inside the installer: "
            f"{sorted(unvetted)}\n"
            "Add each name to _DIST_ENV_DENY (preferred) or, if the app truly needs it "
            "at runtime, to _DIST_ENV_KNOWN_SHIPPED in build_exe.py."
        )

    if dropped:
        print(f"  [env] stripped {len(dropped)} server-side secret(s): {sorted(dropped)}")

    try:
        target.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    except Exception:
        return _stage_runtime_file(src, staging_dir)
    return target


def _add_data_args(project_root: Path, build_root: Path) -> list[str]:
    sep = _data_sep()
    staging_dir = build_root / "_staging"
    pairs: list[tuple[Path, str]] = []

    # Assets used by PDF/report UI
    pairs.append((project_root / "assets", "assets"))

    # Config containing clinical_config.yaml
    pairs.append((project_root / "config", "config"))

    # Runtime config/demo files often expected in working directory.
    # IMPORTANT: Machine-specific data files (users.json, ecg_auth_session.json,
    # ecg_settings.json) must NEVER be bundled from the dev machine into the
    # distributed installer — doing so copies registration records from one machine
    # into every new installation, causing "Machine already registered" errors on
    # the second device that tries to sign up with the same setup file.
    # Always use clean empty placeholders for these files.
    MACHINE_SPECIFIC_FILES = {"users.json", "ecg_auth_session.json"}
    for filename in [
        ".env",
        "customer_channels.json",
        "dummycsv.csv",
        "users.json",
        "ecg_settings.json",
        "last_conclusions.json",
        "ecg_auth_session.json",
        "Animation - 1777012518993.gif",
    ]:
        if filename == ".env":
            file_path = project_root / filename
            staged = _stage_env_for_distribution(file_path, staging_dir)
        elif filename in MACHINE_SPECIFIC_FILES:
            # Always create a clean empty placeholder — never copy real data from dev machine
            staged = _stage_json_placeholder(staging_dir, filename)
        else:
            file_path = project_root / filename
            staged = _stage_runtime_file(file_path, staging_dir)
        if staged is None and filename.endswith(".json"):
            staged = _stage_json_placeholder(staging_dir, filename)
        if staged is not None:
            pairs.append((staged, "."))

    # Some deployments use src-side settings files as fallbacks.
    # Same rule: ecg_auth_session.json and users.json must always be empty placeholders.
    SRC_MACHINE_SPECIFIC = {"ecg_auth_session.json", "users.json"}
    for filename in [
        "src/ecg_settings.json",
        "src/users.json",
        "src/ecg_auth_session.json",
    ]:
        base_name = Path(filename).name
        if base_name in SRC_MACHINE_SPECIFIC:
            # Always clean placeholder — no real session/user data in installer
            staged = _stage_json_placeholder(staging_dir, base_name)
        else:
            file_path = project_root / filename
            staged = _stage_runtime_file(file_path, staging_dir, base_name)
            if staged is None:
                staged = _stage_json_placeholder(staging_dir, base_name)
        if staged is not None:
            pairs.append((staged, "src"))

    args: list[str] = []
    for src, dst in pairs:
        if src.exists():
            args.extend(["--add-data", f"{src}{sep}{dst}"])
    return args


def build_args(project_root: Path, name: str, onefile: bool, console: bool) -> list[str]:
    main_script = project_root / "src" / "main.py"
    if not main_script.exists():
        raise FileNotFoundError(f"Main script not found: {main_script}")

    build_dir = project_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    dist_dir = project_root / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    args: list[str] = [
        str(main_script),
        f"--name={name}",
        "--noconfirm",
        "--clean",
        "--windowed" if not console else "--console",
        "--onedir" if not onefile else "--onefile",
        f"--icon={project_root / 'assets' / 'cardiox_logo.ico'}",
        # FIX: src/ on path so PyInstaller finds organization.py and all src modules
        f"--paths={project_root / 'src'}",
        # FIX: write spec to build/ so stale local spec is never used on CI
        f"--specpath={build_dir}",
        f"--workpath={build_dir / 'work'}",
        f"--distpath={dist_dir}",
        # Imported dynamically via importlib in src/main.py
        "--hidden-import=organization",
        # `importlib` is stdlib, but include explicitly for deterministic PyInstaller analysis.
        "--hidden-import=importlib",
        "--hidden-import=serial",
        "--hidden-import=serial.tools.list_ports",
        "--hidden-import=serial.tools.list_ports.windows",
        "--hidden-import=serial.tools.list_ports.posix",
        # ECG modules imported dynamically
        "--hidden-import=ecg.twelve_lead_test",
        "--hidden-import=ecg.expanded_lead_view",
        "--hidden-import=ecg.hrv_test",
        "--hidden-import=ecg.hyperkalemia_test",
        "--hidden-import=ecg.ecg_report_generator",
        "--hidden-import=ecg.clinical_measurements",
        "--hidden-import=ecg.arrhythmia_detector",
        "--hidden-import=ecg.pan_tompkins",
        "--hidden-import=ecg.recording",
        "--hidden-import=ecg.demo_manager",
        "--hidden-import=ecg.signal_paths",
        "--hidden-import=ecg.ecg_calculations",
        "--hidden-import=ecg.ecg_filters",
        "--hidden-import=ecg.clinical_validation",
        "--collect-submodules=ecg",
        "--collect-submodules=ecg.serial",
        "--collect-submodules=ecg.signal",
        "--collect-submodules=ecg.metrics",
        "--collect-submodules=ecg.plotting",
        "--collect-submodules=ecg.ui",
        "--collect-submodules=ecg.utils",
        "--collect-submodules=ecg.holter",
        "--collect-submodules=ecg.arrhythmia_engine",
        "--collect-submodules=dashboard",
        "--collect-submodules=utils",
        "--collect-submodules=auth",
        "--collect-submodules=config",
        "--collect-submodules=core",
        "--hidden-import=PyQt5",
        "--hidden-import=PyQt5.sip",
        "--hidden-import=PyQt5.QtNetwork",
        "--hidden-import=fitz",
        "--hidden-import=pymupdf",
        # PyYAML is required by clinical_config.py to load arrhythmia thresholds.
        # Without this, the frozen app silently uses Python defaults and arrhythmia
        # detection produces "Rhythm Undetermined" instead of the correct diagnosis.
        "--hidden-import=yaml",
        "--collect-all=PyYAML",
        "--collect-submodules=PyQt5",
        "--collect-all=pyqtgraph",
        "--collect-all=scipy",
        "--exclude-module=PyQt6",
        "--exclude-module=PySide2",
        "--exclude-module=PySide6",
        "--exclude-module=notebook",
        "--exclude-module=jinja2",
        "--exclude-module=IPython",
    ]
    args.extend(_add_data_args(project_root, build_dir))
    return args


def main() -> int:
    _encode_stdio_utf8_on_windows()

    parser = argparse.ArgumentParser(description="Build CardioX executable with PyInstaller")
    parser.add_argument("--name", default="CardioX", help="Output application name")
    parser.add_argument("--onefile", action="store_true", help="Build as onefile (not recommended)")
    parser.add_argument("--console", action="store_true", help="Build with console for debugging")
    parser.add_argument(
        "--admin",
        action="store_true",
        help="Request UAC admin privileges on launch (adds PyInstaller --uac-admin; Windows only).",
    )
    parser.add_argument(
        "--runtime-tmpdir",
        default=None,
        help=(
            "Where PyInstaller onefile extracts runtime files. "
            "Example (Windows): TEMP\\CardioX (use your TEMP environment variable when running the command). "
            "If omitted, PyInstaller default is used."
        ),
    )
    ns = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    args = build_args(project_root, ns.name, ns.onefile, ns.console)

    # Only relevant for onefile builds. If not set, PyInstaller will use its default temp location.
    if ns.onefile and ns.runtime_tmpdir:
        runtime_tmpdir_raw = ns.runtime_tmpdir.strip()
        # IMPORTANT:
        # - PyInstaller embeds --runtime-tmpdir into the executable at build time.
        # - Using %TEMP% / $env:TEMP would get expanded on the build machine and break on other PCs.
        if "%" in runtime_tmpdir_raw or "$env:" in runtime_tmpdir_raw or "$" in runtime_tmpdir_raw:
            raise SystemExit(
                "ERROR: --runtime-tmpdir must be an absolute path and must not contain environment variables "
                "(e.g. %TEMP% or $env:TEMP). Omit --runtime-tmpdir to use the per-user temp directory on each PC."
            )
        runtime_tmpdir = Path(runtime_tmpdir_raw)
        if not runtime_tmpdir.is_absolute():
            raise SystemExit(
                "ERROR: --runtime-tmpdir must be an absolute path (e.g. C:\\Temp\\CardioX). "
                "Omit --runtime-tmpdir to use the per-user temp directory on each PC."
            )
        args.insert(5, f"--runtime-tmpdir={runtime_tmpdir}")

    print("=" * 70)
    print("Building CardioX executable")
    print(f"Project root : {project_root}")
    print(f"Mode         : {'onefile' if ns.onefile else 'onedir (recommended)'}")
    print(f"Console      : {'ON' if ns.console else 'OFF'}")
    print(f"Admin        : {'ON' if ns.admin else 'OFF'}")
    print("=" * 70)
    print("PyInstaller args:")
    for a in args:
        print(f"  {a}")

    try:
        if ns.admin:
            args.append("--uac-admin")
        PyInstaller.__main__.run(args)
    except Exception as exc:
        print(f"\nBuild failed: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    if ns.onefile:
        print(f"\nBuild complete: dist/{ns.name}.exe")
    else:
        print(f"\nBuild complete: dist/{ns.name}/{ns.name}.exe")
        print("Distribute the entire dist folder for this app (not exe alone).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
