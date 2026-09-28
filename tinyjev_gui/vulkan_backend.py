from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np

from .vulkan_setup import find_llama_server, llama_has_vulkan, preferred_gguf

_kill_job = None


def _bind_to_this_process(process: subprocess.Popen) -> None:
    global _kill_job
    if not sys.platform.startswith("win"):
        return

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.OpenProcess.restype = wintypes.HANDLE

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_ulonglong)
            for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )
        ]

    class BASIC_LIMIT(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BASIC_LIMIT),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    if _kill_job is None:
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return
        info = EXTENDED_LIMIT()
        info.BasicLimitInformation.LimitFlags = 0x2000
        if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            kernel32.CloseHandle(job)
            return
        _kill_job = job

    handle = kernel32.OpenProcess(0x0001 | 0x0100, False, process.pid)
    if handle:
        kernel32.AssignProcessToJobObject(_kill_job, handle)
        kernel32.CloseHandle(handle)


def _find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _extract_embeddings(payload: Any) -> np.ndarray:
    data = payload
    if isinstance(payload, dict) and "data" in payload:
        data = payload["data"]

    if isinstance(data, list) and data:
        item = data[0]
    elif isinstance(data, dict):
        item = data
    else:
        raise ValueError(f"Unexpected embedding payload type: {type(payload).__name__}")

    vectors = item.get("embedding") or item.get("embeddings")
    if vectors is None:
        raise ValueError("embedding payload has no embedding field")

    arr = np.array(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


class LlamaServerHandle:
    def __init__(
        self,
        executable: Path,
        gguf_path: Path,
        host: str,
        port: int,
        mode: Literal["vulkan", "cpu"],
        process: subprocess.Popen[str],
    ):
        self.executable = executable
        self.gguf_path = gguf_path
        self.host = host
        self.port = port
        self.mode = mode
        self.process = process
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            text = line.rstrip()
            with self._lock:
                self._lines.append(text)
                if len(self._lines) > 5000:
                    self._lines = self._lines[-5000:]

    def log_tail(self, max_lines: int = 80) -> str:
        with self._lock:
            lines = self._lines[-max_lines:]
        return "\n".join(lines)

    def gpu_memory_lines(self) -> list[str]:
        with self._lock:
            lines = list(self._lines)
        out = []
        for line in lines:
            low = line.lower()
            if "vulkan" in low and "mib" in low:
                out.append(line)
        return out

    def wait_ready(self, timeout_s: float = 120.0) -> None:
        deadline = time.time() + timeout_s
        url = f"{self.base_url}/health"
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited early\n{self.log_tail()}")
            try:
                response = httpx.get(url, timeout=0.8)
                if response.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.25)
        raise RuntimeError(f"Timed out waiting for llama-server at {url}\n{self.log_tail()}")

    def stop(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=8)

    @classmethod
    def start(
        cls,
        model_name: str,
        gguf_path: Path | None = None,
        host: str = "127.0.0.1",
        port: int | None = None,
        mode: Literal["vulkan", "cpu"] = "vulkan",
    ) -> "LlamaServerHandle":
        executable = find_llama_server(auto_download=True)
        if mode == "vulkan" and not llama_has_vulkan(str(executable)):
            raise RuntimeError(f"{executable} does not report a Vulkan device")

        model_path = gguf_path or preferred_gguf(model_name)
        if not model_path.exists():
            raise FileNotFoundError(f"Missing GGUF: {model_path}")

        selected_port = int(port or os.environ.get("TINYJEV_LLAMA_PORT", "0") or 0)
        if selected_port <= 0:
            selected_port = _find_free_port(host)

        ctx_size = int(os.environ.get("TINYJEV_LLAMA_CTX", "8192"))
        batch = int(os.environ.get("TINYJEV_LLAMA_BATCH", str(ctx_size)))

        if mode == "cpu":
            ngl = int(os.environ.get("TINYJEV_LLAMA_NGL", "0"))
        else:
            ngl = int(os.environ.get("TINYJEV_LLAMA_NGL", "99"))

        cmd = [
            str(executable),
            "-m",
            str(model_path),
            "--host",
            host,
            "--port",
            str(selected_port),
            "--ctx-size",
            str(ctx_size),
            "--embeddings",
            "--pooling",
            "none",
            "--parallel",
            "1",
            "-ngl",
            str(ngl),
            "-b",
            str(batch),
            "-ub",
            str(batch),
        ]
        if mode == "vulkan":
            cmd.extend(["--flash-attn", "on"])

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        _bind_to_this_process(process)
        handle = cls(
            executable=executable,
            gguf_path=model_path,
            host=host,
            port=selected_port,
            mode=mode,
            process=process,
        )
        try:
            handle.wait_ready(timeout_s=180)
        except Exception:
            handle.stop()
            raise
        return handle


_shared_handle: LlamaServerHandle | None = None
_shared_lock = threading.Lock()


def get_or_start_shared_server(model_name: str, mode: Literal["vulkan", "cpu"] = "vulkan") -> LlamaServerHandle:
    global _shared_handle
    with _shared_lock:
        if _shared_handle and _shared_handle.process.poll() is None and _shared_handle.mode == mode:
            return _shared_handle
        if _shared_handle:
            _shared_handle.stop()
        _shared_handle = LlamaServerHandle.start(model_name=model_name, mode=mode)
        return _shared_handle


def stop_shared_server() -> None:
    global _shared_handle
    with _shared_lock:
        if _shared_handle:
            _shared_handle.stop()
            _shared_handle = None


class LlamaCppQwen3Backbone:
    name = "llama.cpp"

    def __init__(self, config: dict, weights_path: str, prefix_min_tokens: int = 96, device=None):
        self.config = config
        self.weights_path = Path(weights_path)
        self.prefix_min_tokens = prefix_min_tokens
        self._owned_server: LlamaServerHandle | None = None

        base_url = os.environ.get("TINYJEV_LLAMA_URL", "").strip()
        mode = os.environ.get("TINYJEV_LLAMA_MODE", "vulkan").strip().lower()
        if mode not in {"vulkan", "cpu"}:
            mode = "vulkan"

        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            model_name = os.environ.get("TINYJEV_MODEL_NAME", "TinyJev-0.6B")
            server = get_or_start_shared_server(model_name=model_name, mode=mode)
            self._owned_server = server
            self.base_url = server.base_url

        self.client = httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))

    def _row_hidden(self, token_ids: list[int]) -> np.ndarray:
        payload = {"content": token_ids, "add_special": False}
        response = self.client.post(f"{self.base_url}/embedding", json=payload)
        response.raise_for_status()
        return _extract_embeddings(response.json())

    def hidden_rows(self, prefix, suffixes, pad):
        rows: list[np.ndarray] = []
        for suffix in suffixes:
            token_ids = list(prefix) + list(suffix)
            rows.append(self._row_hidden(token_ids))
        return rows

    def shutdown(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

        owned = self._owned_server
        if owned and os.environ.get("TINYJEV_LLAMA_URL", "").strip() == "":
            stop_shared_server()
            self._owned_server = None


VulkanQwen3Backbone = LlamaCppQwen3Backbone
