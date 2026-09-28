import json
import os
import subprocess
import threading
from pathlib import Path

import httpx
from huggingface_hub import HfApi, constants as hf_constants, snapshot_download
from PySide6.QtCore import QThread, Signal

from .gpu_setup import ensure_gpu_runtime
from .vulkan_setup import convert_model_to_gguf


class RequestWorker(QThread):
    done = Signal(str)

    def __init__(self, url, payload):
        super().__init__()
        self.url = url
        self.payload = payload

    def run(self):
        try:
            response = httpx.post(self.url, json=self.payload, timeout=120)
            try:
                data = response.json()
                if response.status_code >= 400:
                    wrapped = {"status": response.status_code, "error": data}
                    self.done.emit(json.dumps(wrapped, indent=2))
                else:
                    self.done.emit(json.dumps(data, indent=2))
            except ValueError:
                self.done.emit(f"HTTP {response.status_code}\n{response.text}")
        except Exception as exc:
            self.done.emit(f"Request failed: {exc}")


class HealthWorker(QThread):
    done = Signal(bool)

    def __init__(self, url):
        super().__init__()
        self.url = url

    def run(self):
        ready = False
        try:
            response = httpx.get(self.url, timeout=1.2)
            if response.status_code == 200:
                try:
                    body = response.json()
                    ready = bool(body.get("ready", True))
                except ValueError:
                    ready = True
        except Exception:
            ready = False
        self.done.emit(ready)


class ModelMetadataWorker(QThread):
    done = Signal(object, object)

    def __init__(self, entries, timeout_seconds=2.5):
        super().__init__()
        self.entries = [(str(name), str(repo_id)) for name, repo_id in entries]
        self.timeout_seconds = max(0.5, float(timeout_seconds))

    def fetch_with_timeout(self, api, repo_id):
        result = {}

        def target():
            try:
                result["info"] = api.model_info(repo_id, files_metadata=True)
            except Exception:
                result["info"] = None

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(self.timeout_seconds)
        if thread.is_alive():
            return None
        return result.get("info")

    def run(self):
        api = HfApi()
        expected_files = {}
        expected_sizes = {}

        for model_name, repo_id in self.entries:
            info = self.fetch_with_timeout(api, repo_id)
            if info is None:
                continue

            files = [item.rfilename for item in info.siblings if getattr(item, "rfilename", None)]
            sizes = [item.size for item in info.siblings if getattr(item, "size", None)]
            expected_files[model_name] = files
            if sizes:
                expected_sizes[model_name] = int(sum(sizes))

        self.done.emit(expected_files, expected_sizes)


class DownloadWorker(QThread):
    started_model = Signal(str, object, object)
    progress = Signal(str, object, object, float)
    finished_model = Signal(str, bool, str)

    def __init__(self, model_name, repo_id, initial_bytes=0, total_hint=0, allow_patterns=None):
        super().__init__()
        self.model_name = model_name
        self.repo_id = repo_id
        self.initial_bytes = max(0, int(initial_bytes))
        self.total_hint = max(0, int(total_hint))
        self.allow_patterns = [str(item) for item in (allow_patterns or [])]
        self._cancel_event = threading.Event()
        self._downloaded_delta = 0

    def cancel(self):
        self._cancel_event.set()

    def run(self):
        total_bytes = self.total_hint
        patterns = set(self.allow_patterns)
        try:
            api = HfApi()
            info = api.model_info(self.repo_id, files_metadata=True)
            sized = []
            for entry in info.siblings:
                size = getattr(entry, "size", None)
                if not size:
                    continue
                name = getattr(entry, "rfilename", "")
                if patterns and name not in patterns:
                    continue
                sized.append(int(size))
            if sized:
                total_bytes = int(sum(sized))
        except Exception:
            pass

        self.started_model.emit(self.model_name, self.initial_bytes, total_bytes)

        outer = self
        cancel_event = self._cancel_event

        try:
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            hf_constants.HF_HUB_DISABLE_XET = True

            from tqdm.auto import tqdm

            class ProgressTqdm(tqdm):
                def __init__(self, *args, **kwargs):
                    kwargs["disable"] = True
                    super().__init__(*args, **kwargs)

                def update(self, n=1):
                    if cancel_event.is_set():
                        raise RuntimeError("DOWNLOAD_CANCELLED")
                    before = self.n
                    result = super().update(n)
                    delta = int(self.n - before)
                    if delta > 0:
                        outer._downloaded_delta += delta
                        rate = self.format_dict.get("rate") or 0.0
                        outer.progress.emit(
                            outer.model_name,
                            outer.initial_bytes + outer._downloaded_delta,
                            total_bytes,
                            float(rate),
                        )
                    return result

            snapshot_download(
                repo_id=self.repo_id,
                local_files_only=False,
                max_workers=1,
                tqdm_class=ProgressTqdm,
                allow_patterns=self.allow_patterns or None,
            )

            final_bytes = max(self.initial_bytes + self._downloaded_delta, total_bytes)
            self.progress.emit(self.model_name, final_bytes, total_bytes, 0.0)
            self.finished_model.emit(self.model_name, False, "")
        except Exception as exc:
            if self._cancel_event.is_set() or "DOWNLOAD_CANCELLED" in str(exc):
                self.finished_model.emit(self.model_name, True, "")
            else:
                self.finished_model.emit(self.model_name, False, str(exc))


class GpuRuntimeSetupWorker(QThread):
    log = Signal(str)
    done = Signal(bool, str)

    def __init__(self, runtime_dir, repo_root):
        super().__init__()
        self.runtime_dir = Path(runtime_dir)
        self.repo_root = Path(repo_root)
        self._cancel_event = threading.Event()
        self._process_lock = threading.Lock()
        self._active_process = None

    def _set_active_process(self, process):
        with self._process_lock:
            self._active_process = process

    def _kill_active_process(self):
        with self._process_lock:
            process = self._active_process
        if process is None or process.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                process.kill()
        except Exception:
            pass

    def cancel(self):
        self._cancel_event.set()
        self._kill_active_process()

    def run(self):
        try:
            py = ensure_gpu_runtime(
                runtime_dir=self.runtime_dir,
                repo_root=self.repo_root,
                log=lambda msg: self.log.emit(str(msg)),
                cancel_event=self._cancel_event,
                on_process_start=self._set_active_process,
            )
        except Exception as exc:
            if self._cancel_event.is_set() and "OPERATION_CANCELLED" in str(exc):
                self.done.emit(False, "Cancelled")
            else:
                self.done.emit(False, str(exc))
            return
        finally:
            self._set_active_process(None)

        self.done.emit(True, str(py))


class GgufConvertWorker(QThread):
    log = Signal(str)
    done = Signal(bool, str)

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self._cancel_event = threading.Event()
        self._process_lock = threading.Lock()
        self._active_process = None

    def _set_active_process(self, process):
        with self._process_lock:
            self._active_process = process

    def _kill_active_process(self):
        with self._process_lock:
            process = self._active_process
        if process is None or process.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                process.kill()
        except Exception:
            pass

    def cancel(self):
        self._cancel_event.set()
        self._kill_active_process()

    def run(self):
        try:
            outputs = convert_model_to_gguf(
                self.model_name,
                log=lambda msg: self.log.emit(str(msg)),
                cancel_event=self._cancel_event,
                on_process_start=self._set_active_process,
            )
        except Exception as exc:
            if self._cancel_event.is_set() and "OPERATION_CANCELLED" in str(exc):
                self.done.emit(False, "Cancelled")
            else:
                self.done.emit(False, str(exc))
            return
        finally:
            self._set_active_process(None)

        text = ", ".join(str(path) for path in outputs)
        self.done.emit(True, text)

