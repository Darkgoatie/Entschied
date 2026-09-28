from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

APP_VERSION = "0.1.0"


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
    candidates = [
        Path("C:/Program Files (x86)/Inno Setup 6/ISCC.exe"),
        Path("C:/Program Files/Inno Setup 6/ISCC.exe"),
        Path("C:/Users/halit/AppData/Local/Programs/Inno Setup 6/ISCC.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> int:
    repo = Path(__file__).resolve().parent
    scratch = Path("C:/Users/halit/AppData/Local/hermes/cache/scratch/tinyjev_build")
    scratch.mkdir(parents=True, exist_ok=True)

    python = repo / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        raise FileNotFoundError(f"Missing venv python: {python}")

    llama_server = repo / "app" / "llama" / "llama-server.exe"
    if not llama_server.exists():
        raise FileNotFoundError(f"Bundled llama.cpp is missing: {llama_server}")

    run([str(python), "-m", "pip", "install", "--upgrade", "pip"], repo)
    run([str(python), "-m", "pip", "install", "pyinstaller"], repo)

    build_dir = repo / "build"
    dist_dir = repo / "dist"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    if dist_dir.exists():
        shutil.rmtree(dist_dir)

    run([str(python), "-m", "PyInstaller", "--noconfirm", "--clean", "TinyJev.spec"], repo)

    tinyjev_exe = repo / "dist" / "TinyJev" / "TinyJev.exe"
    if not tinyjev_exe.exists():
        raise FileNotFoundError(f"Build did not produce {tinyjev_exe}")

    smoke_test_frozen(tinyjev_exe, repo)

    iscc = find_iscc()
    if iscc is None:
        run(["winget", "install", "-e", "--id", "JRSoftware.InnoSetup"], repo)
        iscc = find_iscc()
    if iscc is None:
        raise FileNotFoundError("Inno Setup 6 (ISCC.exe) not found after winget install")

    run([str(iscc), "TinyJevInstaller.iss"], repo)

    installer = repo / "dist" / f"TinyJev-Setup-{APP_VERSION}.exe"
    if not installer.exists():
        raise FileNotFoundError(f"Installer not found: {installer}")

    print(f"Built installer: {installer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
