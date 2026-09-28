from __future__ import annotations

import json
import os
import sys
import shutil
import subprocess
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Callable

from huggingface_hub import snapshot_download, try_to_load_from_cache

from .runtime import app_root, is_frozen_app

LLAMA_COMMIT = "5266f24da"
LLAMA_BUNDLED_TAG = "b11236"
LLAMA_BUNDLED_ZIP = f"llama-{LLAMA_BUNDLED_TAG}-bin-win-vulkan-x64.zip"
DIRECTML_SIZE_LIMIT_BYTES = int(5.5 * 1024**3)
GGUF_SUPPORT_FILES = [
    "head.safetensors",
    "tinyjev.json",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "README.md",
]
GGUF_SOURCES = {
    "TinyJev-0.6B": {
        "repo_id": "darkgoatie/TinyJev-0.6B-GGUF",
        "default_quant": "f16",
        "quants": {
            "f16": "TinyJev-0.6B-f16.gguf",
        },
    },
    "TinyJev-4B": {
        "repo_id": "darkgoatie/TinyJev-4B-GGUF",
        "default_quant": "f16",
        "quants": {
            "f16": "TinyJev-4B-F16.gguf",
            "q8_0": "TinyJev-4B-Q8_0.gguf",
        },
    },
}


def _log_default(message: str) -> None:
    print(message, flush=True)


def local_appdata() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", Path.home()))


def gguf_dir() -> Path:
    override = os.environ.get("TINYJEV_GGUF_DIR", "").strip()
    if override:
        return Path(override)
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


def gguf_repo_id(model_name: str) -> str:
    source = GGUF_SOURCES.get(model_name)
    if source is None:
        raise KeyError(f"No GGUF source configured for {model_name}")
    return str(source["repo_id"])


def gguf_quants(model_name: str) -> list[str]:
    source = GGUF_SOURCES.get(model_name)
    if source is None:
        return []
    quants = source.get("quants", {})
    return [str(item) for item in quants]


def default_gguf_quant(model_name: str) -> str:
    source = GGUF_SOURCES.get(model_name)
    if source is None:
        return "f16"
    return str(source.get("default_quant", "f16"))


def resolve_gguf_quant(model_name: str, preferred_quant: str | None = None) -> str:
    quants = gguf_quants(model_name)
    wanted = []
    if preferred_quant:
        wanted.append(preferred_quant)

    env_quant = os.environ.get("TINYJEV_GGUF_QUANT", "").strip()
    if env_quant:
        wanted.append(env_quant)

    wanted.append(default_gguf_quant(model_name))

    for quant in wanted:
        if quant in quants:
            return quant
    if quants:
        return quants[0]
    return "f16"


def gguf_filename(model_name: str, quant: str) -> str:
    source = GGUF_SOURCES.get(model_name)
    if source is None:
        raise KeyError(f"No GGUF source configured for {model_name}")
    quants = source.get("quants", {})
    filename = quants.get(quant)
    if filename is None:
        raise KeyError(f"No GGUF quant {quant} for {model_name}")
    return str(filename)


def gguf_allow_patterns(model_name: str, preferred_quant: str | None = None) -> list[str]:
    quant = resolve_gguf_quant(model_name, preferred_quant)
    return [gguf_filename(model_name, quant), *GGUF_SUPPORT_FILES]


def cached_gguf_file(model_name: str, quant: str) -> Path | None:
    try:
        cached = try_to_load_from_cache(
            repo_id=gguf_repo_id(model_name),
            filename=gguf_filename(model_name, quant),
            repo_type="model",
        )
    except Exception:
        return None

    if isinstance(cached, str):
        path = Path(cached)
        if path.exists():
            return path
    return None


def cached_gguf_bundle(model_name: str, preferred_quant: str | None = None) -> dict[str, object]:
    quant = resolve_gguf_quant(model_name, preferred_quant)
    filenames = gguf_allow_patterns(model_name, quant)
    files: dict[str, Path] = {}
    total_bytes = 0
    for filename in filenames:
        try:
            cached = try_to_load_from_cache(
                repo_id=gguf_repo_id(model_name),
                filename=filename,
                repo_type="model",
            )
        except Exception:
            cached = None
        if isinstance(cached, str):
            path = Path(cached)
            if path.exists():
                files[filename] = path
                total_bytes += path.stat().st_size

    return {
        "model": model_name,
        "quant": quant,
        "repo_id": gguf_repo_id(model_name),
        "files": files,
        "bytes": total_bytes,
        "ready": len(files) == len(filenames),
        "found": len(files),
        "expected": len(filenames),
    }


def resolve_gguf_snapshot(
    model_name: str,
    preferred_quant: str | None = None,
    local_files_only: bool = True,
) -> tuple[Path, Path, str]:
    quant = resolve_gguf_quant(model_name, preferred_quant)
    snapshot = Path(
        snapshot_download(
            repo_id=gguf_repo_id(model_name),
            local_files_only=local_files_only,
            allow_patterns=gguf_allow_patterns(model_name, quant),
        )
    )
    gguf_path = snapshot / gguf_filename(model_name, quant)
    if not gguf_path.exists():
        raise FileNotFoundError(f"Missing {gguf_path.name} in snapshot for {model_name}")
    return snapshot, gguf_path, quant


def gguf_status(model_name: str) -> list[dict[str, object]]:
    rows = []
    for quant, local_path in gguf_targets(model_name):
        cache_path = cached_gguf_file(model_name, quant)
        if cache_path is not None:
            rows.append(
                {
                    "quant": quant,
                    "source": "downloaded",
                    "path": cache_path,
                    "local_path": local_path,
                    "exists": True,
                    "bytes": cache_path.stat().st_size,
                }
            )
            continue

        if local_path.exists():
            rows.append(
                {
                    "quant": quant,
                    "source": "converted",
                    "path": local_path,
                    "local_path": local_path,
                    "exists": True,
                    "bytes": local_path.stat().st_size,
                }
            )
            continue

        rows.append(
            {
                "quant": quant,
                "source": "not_downloaded",
                "path": local_path,
                "local_path": local_path,
                "exists": False,
                "bytes": 0,
            }
        )
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


def bundled_llama_server() -> Path | None:
    roots = [app_root() / "llama", app_root() / "app" / "llama"]
    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir:
        roots.insert(0, Path(bundle_dir) / "llama")
    for root in roots:
        direct = root / "llama-server.exe"
        if direct.exists():
            return direct
        found = _search_llama_server(root)
        if found is not None:
            return found
    return None


def _atomic_llama_server() -> Path:
    appdata = Path(os.environ.get("APPDATA", local_appdata()))
    return (
        appdata
        / "Atomic Chat"
        / "data"
        / "llamacpp-upstream"
        / "backends"
        / "b10809"
        / "win-vulkan-x64"
        / "build"
        / "bin"
        / "llama-server.exe"
    )


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


def _find_vulkan_asset(repo: str) -> tuple[str, dict[str, object]]:
    for page in range(1, 11):
        url = f"https://api.github.com/repos/{repo}/releases?per_page=100&page={page}"
        with urllib.request.urlopen(url, timeout=30) as response:
            releases = json.loads(response.read().decode("utf-8"))
        if not releases:
            break

        for release in releases:
            assets = release.get("assets", [])
            for asset in assets:
                name = str(asset.get("name", "")).lower()
                if "win-vulkan-x64" in name and name.endswith(".zip"):
                    return str(release.get("tag_name", "unknown")), asset

    raise RuntimeError(f"No Windows Vulkan zip found in releases for {repo}")


def _download_latest_vulkan_release(log: Callable[[str], None]) -> Path:
    root = llama_dir()
    downloads = root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    repos = ["ggml-org/llama.cpp", "ggerganov/llama.cpp"]
    errors: list[str] = []

    for repo in repos:
        try:
            tag_name, selected = _find_vulkan_asset(repo)
            zip_name = str(selected["name"])
            zip_url = str(selected["browser_download_url"])
            zip_path = downloads / zip_name
            extract_dir = root / tag_name

            if not zip_path.exists():
                log(f"Downloading {zip_name} from {repo} ({tag_name})...")
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

    bundled = bundled_llama_server()
    if bundled is not None:
        return bundled

    if not is_frozen_app():
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

    if is_frozen_app():
        raise FileNotFoundError("Bundled llama-server.exe not found in frozen app")

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


def _run_logged(
    cmd: list[str],
    log: Callable[[str], None],
    cwd: Path | None = None,
    cancel_event=None,
    on_process_start: Callable[[subprocess.Popen | None], None] | None = None,
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("OPERATION_CANCELLED")

    log(f"> {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if on_process_start is not None:
        on_process_start(proc)

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if cancel_event is not None and cancel_event.is_set():
                proc.kill()
                proc.wait()
                raise RuntimeError("OPERATION_CANCELLED")
            log(line.rstrip())
        code = proc.wait()
    finally:
        if on_process_start is not None:
            on_process_start(None)

    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("OPERATION_CANCELLED")

    if code != 0:
        raise RuntimeError(f"Command failed ({code}): {' '.join(cmd)}")


def ensure_convert_tooling(
    log: Callable[[str], None] | None = None,
    cancel_event=None,
    on_process_start: Callable[[subprocess.Popen | None], None] | None = None,
) -> tuple[Path, Path]:
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

    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("OPERATION_CANCELLED")

    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        _run_logged(
            ["git", "clone", "--filter=blob:none", "https://github.com/ggerganov/llama.cpp", str(repo_dir)],
            log,
            cancel_event=cancel_event,
            on_process_start=on_process_start,
        )

    _run_logged(
        ["git", "checkout", LLAMA_COMMIT],
        log,
        cwd=repo_dir,
        cancel_event=cancel_event,
        on_process_start=on_process_start,
    )

    if not py.exists():
        _run_logged(
            ["python", "-m", "venv", str(venv_dir)],
            log,
            cancel_event=cancel_event,
            on_process_start=on_process_start,
        )

    _run_logged(
        [str(py), "-m", "pip", "install", "--upgrade", "pip"],
        log,
        cancel_event=cancel_event,
        on_process_start=on_process_start,
    )
    _run_logged(
        [str(py), "-m", "pip", "install", "-r", str(repo_dir / "requirements" / "requirements-convert_hf_to_gguf.txt")],
        log,
        cancel_event=cancel_event,
        on_process_start=on_process_start,
    )
    _run_logged(
        [str(py), "-m", "pip", "install", "gguf"],
        log,
        cancel_event=cancel_event,
        on_process_start=on_process_start,
    )

    return py, script


def convert_model_to_gguf(
    model_name: str,
    log: Callable[[str], None] | None = None,
    cancel_event=None,
    on_process_start: Callable[[subprocess.Popen | None], None] | None = None,
) -> list[Path]:
    log = log or _log_default

    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("OPERATION_CANCELLED")

    snapshot = resolve_snapshot(model_name, local_files_only=True)
    if not snapshot.exists():
        raise FileNotFoundError(f"Model files for {model_name} are not available locally.")

    py, script = ensure_convert_tooling(
        log,
        cancel_event=cancel_event,
        on_process_start=on_process_start,
    )
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
        _run_logged(
            cmd,
            log,
            cancel_event=cancel_event,
            on_process_start=on_process_start,
        )
        outputs.append(output)

    return outputs
