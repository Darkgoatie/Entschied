from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Callable

from huggingface_hub import snapshot_download

LLAMA_COMMIT = "5266f24da"
DIRECTML_SIZE_LIMIT_BYTES = int(5.5 * 1024**3)


def _log_default(message: str) -> None:
    print(message, flush=True)


def local_appdata() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", Path.home()))


def gguf_dir() -> Path:
    return local_appdata() / "TinyJev" / "gguf"


def llama_dir() -> Path:
    return local_appdata() / "TinyJev" / "llama.cpp"


def gguf_targets(model_name: str) -> list[tuple[str, Path]]:
    root = gguf_dir()
    targets = {
        "TinyJev-0.6B": [("f16", root / "tinyjev-0.6b-f16.gguf")],
        "TinyJev-4B": [
            ("f16", root / "tinyjev-4b-f16.gguf"),
            ("q8_0", root / "tinyjev-4b-q8_0.gguf"),
        ],
    }
    return targets.get(model_name, [])


def gguf_status(model_name: str) -> list[dict[str, object]]:
    rows = []
    for quant, path in gguf_targets(model_name):
        if path.exists():
            rows.append({
                "quant": quant,
                "path": path,
                "exists": True,
                "bytes": path.stat().st_size,
            })
        else:
            rows.append({"quant": quant, "path": path, "exists": False, "bytes": 0})
    return rows


def preferred_gguf(model_name: str, preferred_quant: str | None = None) -> Path:
    wanted = []
    if preferred_quant:
        wanted.append(preferred_quant)
    env_quant = os.environ.get("TINYJEV_GGUF_QUANT", "").strip()
    if env_quant:
        wanted.append(env_quant)
    wanted.extend(["f16", "q8_0", "bf16", "f32"])

    targets = {quant: path for quant, path in gguf_targets(model_name)}
    for quant in wanted:
        path = targets.get(quant)
        if path and path.exists():
            return path

    expected = ", ".join(str(path) for path in targets.values())
    raise FileNotFoundError(f"No GGUF found for {model_name}. Expected one of: {expected}")


def model_repo_id(model_name: str) -> str:
    mapping = {
        "TinyJev-0.6B": "AnkitAI/TinyJev-0.6B",
        "TinyJev-4B": "AnkitAI/TinyJev-4B",
    }
    return mapping[model_name]


def resolve_snapshot(model_name: str, local_files_only: bool = True) -> Path:
    return Path(snapshot_download(model_repo_id(model_name), local_files_only=local_files_only))


def model_weights_size(model_name: str) -> int:
    root = resolve_snapshot(model_name, local_files_only=True)
    weight = root / "model.safetensors"
    if not weight.exists():
        return 0
    return weight.stat().st_size


def directml_allowed(model_name: str, limit_bytes: int = DIRECTML_SIZE_LIMIT_BYTES) -> tuple[bool, int]:
    size = model_weights_size(model_name)
    if size <= 0:
        return True, size
    return size <= limit_bytes, size


def _atomic_llama_server() -> Path:
    appdata = Path(os.environ.get("APPDATA", local_appdata()))
    return appdata / "Atomic Chat" / "data" / "llamacpp-upstream" / "backends" / "b10809" / "win-vulkan-x64" / "build" / "bin" / "llama-server.exe"


def _which_llama_server() -> Path | None:
    for candidate in ["llama-server.exe", "llama-server"]:
        found = shutil.which(candidate)
        if found:
            return Path(found)
    return None


def _search_llama_server(root: Path) -> Path | None:
    if not root.exists():
        return None
    for path in root.rglob("llama-server.exe"):
        return path
    return None


@lru_cache(maxsize=4)
def llama_has_vulkan(executable: str) -> bool:
    try:
        proc = subprocess.run(
            [executable, "--list-devices"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return False

    output = (proc.stdout or "") + (proc.stderr or "")
    return "vulkan" in output.lower()


def _download_latest_vulkan_release(log: Callable[[str], None]) -> Path:
    root = llama_dir()
    downloads = root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    repos = ["ggml-org/llama.cpp", "ggerganov/llama.cpp"]
    errors: list[str] = []

    for repo in repos:
        try:
            with urllib.request.urlopen(f"https://api.github.com/repos/{repo}/releases/latest", timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))

            assets = payload.get("assets", [])
            selected = None
            for asset in assets:
                name = str(asset.get("name", "")).lower()
                if "win-vulkan-x64" in name and name.endswith(".zip"):
                    selected = asset
                    break
            if not selected:
                raise RuntimeError(f"No Windows Vulkan zip found in latest release of {repo}")

            zip_name = selected["name"]
            zip_url = selected["browser_download_url"]
            zip_path = downloads / zip_name
            extract_dir = root / payload.get("tag_name", "latest")

            if not zip_path.exists():
                log(f"Downloading {zip_name} from {repo}...")
                urllib.request.urlretrieve(zip_url, zip_path)

            if not extract_dir.exists():
                log(f"Extracting {zip_name}...")
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(extract_dir)

            found = _search_llama_server(extract_dir)
            if found:
                return found
            raise RuntimeError(f"llama-server.exe not found after extracting {zip_name}")
        except Exception as exc:
            errors.append(f"{repo}: {exc}")

    raise RuntimeError("; ".join(errors))


def find_llama_server(log: Callable[[str], None] | None = None, auto_download: bool = True) -> Path:
    log = log or _log_default

    env_path = os.environ.get("TINYJEV_LLAMA_SERVER", "").strip()
    if env_path:
        path = Path(env_path)
        if not path.exists():
            raise FileNotFoundError(f"TINYJEV_LLAMA_SERVER points to missing file: {path}")
        return path

    atomic = _atomic_llama_server()
    if atomic.exists():
        return atomic

    in_local = _search_llama_server(llama_dir())
    if in_local:
        return in_local

    on_path = _which_llama_server()
    if on_path:
        return on_path

    if not auto_download:
        raise FileNotFoundError("llama-server.exe not found")

    return _download_latest_vulkan_release(log)


def _scratch_root() -> Path:
    tmp = os.environ.get("TMPDIR", "")
    if tmp:
        return Path(tmp)
    return local_appdata() / "hermes" / "cache" / "scratch"


def _convert_workspace() -> tuple[Path, Path]:
    root = _scratch_root()
    llama_repo = root / f"llama.cpp-{LLAMA_COMMIT}"
    venv_dir = root / "llama-gguf-venv"
    return llama_repo, venv_dir


def _run_logged(cmd: list[str], log: Callable[[str], None], cwd: Path | None = None) -> None:
    log(f"> {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log(line.rstrip())
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Command failed ({code}): {' '.join(cmd)}")


def ensure_convert_tooling(log: Callable[[str], None] | None = None) -> tuple[Path, Path]:
    log = log or _log_default

    env_py = os.environ.get("TINYJEV_CONVERT_PYTHON", "").strip()
    env_script = os.environ.get("TINYJEV_CONVERT_SCRIPT", "").strip()
    if env_py and env_script:
        py = Path(env_py)
        script = Path(env_script)
        if py.exists() and script.exists():
            return py, script

    repo_dir, venv_dir = _convert_workspace()
    py = venv_dir / "Scripts" / "python.exe"
    script = repo_dir / "convert_hf_to_gguf.py"

    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        _run_logged(["git", "clone", "--filter=blob:none", "https://github.com/ggerganov/llama.cpp", str(repo_dir)], log)

    _run_logged(["git", "checkout", LLAMA_COMMIT], log, cwd=repo_dir)

    if not py.exists():
        _run_logged(["python", "-m", "venv", str(venv_dir)], log)

    _run_logged([str(py), "-m", "pip", "install", "--upgrade", "pip"], log)
    _run_logged([str(py), "-m", "pip", "install", "-r", str(repo_dir / "requirements" / "requirements-convert_hf_to_gguf.txt")], log)
    _run_logged([str(py), "-m", "pip", "install", "gguf"], log)

    return py, script


def convert_model_to_gguf(model_name: str, log: Callable[[str], None] | None = None) -> list[Path]:
    log = log or _log_default

    snapshot = resolve_snapshot(model_name, local_files_only=True)
    if not snapshot.exists():
        raise FileNotFoundError(f"Model files for {model_name} are not available locally.")

    py, script = ensure_convert_tooling(log)
    gguf_dir().mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for quant, output in gguf_targets(model_name):
        if output.exists() and output.stat().st_size > 0:
            log(f"Skipping {output.name} (already exists)")
            outputs.append(output)
            continue

        cmd = [
            str(py),
            str(script),
            str(snapshot),
            "--outtype",
            quant,
            "--outfile",
            str(output),
        ]
        _run_logged(cmd, log)
        outputs.append(output)

    return outputs
