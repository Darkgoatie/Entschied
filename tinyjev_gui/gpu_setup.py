from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path
from typing import Callable


def default_runtime_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    return base / "TinyJev" / "gpu-runtime"


def runtime_python(runtime_dir: Path) -> Path:
    return runtime_dir / "Scripts" / "python.exe"


def run_cmd(
    cmd: list[str],
    log: Callable[[str], None],
    cancel_event=None,
    on_process_start: Callable[[subprocess.Popen | None], None] | None = None,
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("OPERATION_CANCELLED")

    log(f"> {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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
        raise RuntimeError(f"Command failed with exit code {code}: {' '.join(cmd)}")


def ensure_gpu_runtime(
    runtime_dir: Path | None = None,
    repo_root: Path | None = None,
    log: Callable[[str], None] | None = None,
    cancel_event=None,
    on_process_start: Callable[[subprocess.Popen | None], None] | None = None,
) -> Path:
    runtime_dir = Path(runtime_dir or default_runtime_dir())
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1])
    log = log or (lambda msg: print(msg, flush=True))

    py = runtime_python(runtime_dir)
    if not py.exists():
        log(f"Creating runtime at {runtime_dir}")
        venv.EnvBuilder(with_pip=True).create(str(runtime_dir))
    else:
        log(f"Using existing runtime at {runtime_dir}")

    run_cmd(
        [str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        log,
        cancel_event=cancel_event,
        on_process_start=on_process_start,
    )
    run_cmd(
        [str(py), "-m", "pip", "install", "--no-deps", "-e", str(repo_root)],
        log,
        cancel_event=cancel_event,
        on_process_start=on_process_start,
    )
    run_cmd(
        [str(py), "-m", "pip", "install", "tinyjev", "torch-directml", "transformers<5,>=4.40"],
        log,
        cancel_event=cancel_event,
        on_process_start=on_process_start,
    )
    run_cmd(
        [str(py), "-c", "import tinyjev,torch_directml,transformers; print(torch_directml.device())"],
        log,
        cancel_event=cancel_event,
        on_process_start=on_process_start,
    )

    return py


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Create or update TinyJev DirectML runtime")
    parser.add_argument("--runtime-dir", default=str(default_runtime_dir()))
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args(argv)

    runtime_dir = Path(args.runtime_dir)
    repo_root = Path(args.repo_root)

    try:
        py = ensure_gpu_runtime(runtime_dir=runtime_dir, repo_root=repo_root)
    except Exception as exc:
        print(f"GPU runtime setup failed: {exc}", file=sys.stderr, flush=True)
        return 2

    print(f"GPU runtime ready: {py}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
