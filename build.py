from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

APP_VERSION = "0.1.0"
LLAMA_TAG = "b11236"
LLAMA_ZIP = f"llama-{LLAMA_TAG}-bin-win-vulkan-x64.zip"
LLAMA_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/{LLAMA_ZIP}"
VC_RUNTIME_DLLS = (
    "msvcp140.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "concrt140.dll",
)


def run(cmd: list[str], cwd: Path) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def smoke_test_frozen(exe_path: Path, cwd: Path) -> None:
    cmd = [str(exe_path), "--self-test"]
    print(">", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Frozen self-test failed with exit code {result.returncode}")


def find_iscc() -> Path | None:
    local_appdata = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
    candidates = [
        Path("C:/Program Files (x86)/Inno Setup 6/ISCC.exe"),
        Path("C:/Program Files/Inno Setup 6/ISCC.exe"),
        local_appdata / "Programs" / "Inno Setup 6" / "ISCC.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def scratch_dir() -> Path:
    tmpdir = os.environ.get("TMPDIR", "").strip()
    if tmpdir:
        return Path(tmpdir) / "entschied_build"
    local_appdata = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
    return local_appdata / "hermes" / "cache" / "scratch" / "entschied_build"


def _resolve_runtime_dll(name: str) -> Path:
    candidates: list[Path] = []
    vctools = os.environ.get("VCToolsRedistDir", "").strip()
    if vctools:
        base = Path(vctools)
        candidates.extend(
            [
                base / "x64" / "Microsoft.VC143.CRT",
                base / "x64" / "Microsoft.VC142.CRT",
                base / "x64" / "Microsoft.VC141.CRT",
                base / "x64" / "Microsoft.VC140.CRT",
                base,
            ]
        )

    system_root = Path(os.environ.get("SystemRoot", "C:/Windows"))
    candidates.extend([system_root / "System32", system_root / "SysWOW64"])

    for entry in os.environ.get("PATH", "").split(os.pathsep):
        text = entry.strip()
        if text:
            candidates.append(Path(text))

    seen: set[str] = set()
    for folder in candidates:
        key = str(folder).lower()
        if key in seen:
            continue
        seen.add(key)

        try:
            path = folder / name
        except OSError:
            continue

        if path.exists():
            return path

    raise FileNotFoundError(
        f"Missing required runtime DLL: {name}. Install Microsoft Visual C++ Redistributable x64."
    )


def ensure_vc_runtime_dlls(llama_dir: Path) -> None:
    for dll_name in VC_RUNTIME_DLLS:
        target = llama_dir / dll_name
        if target.exists():
            continue
        source = _resolve_runtime_dll(dll_name)
        shutil.copy2(source, target)


def write_bundle_info(llama_dir: Path) -> None:
    info = "\n".join(
        [
            f"llama.cpp tag: {LLAMA_TAG}",
            f"asset: {LLAMA_ZIP}",
            f"source: https://github.com/ggml-org/llama.cpp/releases/tag/{LLAMA_TAG}",
        ]
    )
    (llama_dir / "BUNDLE_INFO.txt").write_text(info + "\n", encoding="utf-8")


def prepare_llama_bundle(repo: Path, scratch: Path) -> None:
    llama_dir = repo / "app" / "llama"
    llama_server = llama_dir / "llama-server.exe"

    if llama_server.exists():
        ensure_vc_runtime_dlls(llama_dir)
        write_bundle_info(llama_dir)
        return

    downloads = scratch / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    zip_path = downloads / LLAMA_ZIP

    if not zip_path.exists():
        print(f"Downloading {LLAMA_ZIP}...")
        urllib.request.urlretrieve(LLAMA_URL, zip_path)

    extract_dir = scratch / f"llama-{LLAMA_TAG}"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {LLAMA_ZIP}...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)

    source_server = next(extract_dir.rglob("llama-server.exe"), None)
    if source_server is None:
        raise FileNotFoundError(f"llama-server.exe not found in {zip_path}")

    source_dir = source_server.parent
    if llama_dir.exists():
        shutil.rmtree(llama_dir)
    llama_dir.mkdir(parents=True, exist_ok=True)

    for item in source_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, llama_dir / item.name)

    if not (llama_dir / "llama-server.exe").exists():
        raise FileNotFoundError("Failed to stage llama-server.exe into app/llama")

    ensure_vc_runtime_dlls(llama_dir)
    write_bundle_info(llama_dir)


def main() -> int:
    repo = Path(__file__).resolve().parent
    scratch = scratch_dir()
    scratch.mkdir(parents=True, exist_ok=True)

    python = repo / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        raise FileNotFoundError(f"Missing venv python: {python}")

    prepare_llama_bundle(repo, scratch)

    run([str(python), "-m", "pip", "install", "--upgrade", "pip"], repo)
    run([str(python), "-m", "pip", "install", "pyinstaller"], repo)

    build_dir = repo / "build"
    dist_dir = repo / "dist"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    if dist_dir.exists():
        shutil.rmtree(dist_dir)

    run([str(python), "-m", "PyInstaller", "--noconfirm", "--clean", "Entschied.spec"], repo)

    app_exe = repo / "dist" / "Entschied" / "Entschied.exe"
    if not app_exe.exists():
        raise FileNotFoundError(f"Build did not produce {app_exe}")

    smoke_test_frozen(app_exe, repo)

    iscc = find_iscc()
    if iscc is None:
        run(["winget", "install", "-e", "--id", "JRSoftware.InnoSetup"], repo)
        iscc = find_iscc()
    if iscc is None:
        raise FileNotFoundError("Inno Setup 6 (ISCC.exe) not found after winget install")

    run([str(iscc), "EntschiedInstaller.iss"], repo)

    installer = repo / "dist" / f"Entschied-Setup-{APP_VERSION}.exe"
    if not installer.exists():
        raise FileNotFoundError(f"Installer not found: {installer}")

    print(f"Built installer: {installer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())