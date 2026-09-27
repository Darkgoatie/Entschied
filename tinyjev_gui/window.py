import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, scan_cache_dir
from huggingface_hub.constants import HF_HUB_CACHE
from PySide6.QtCore import QProcess, QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from tinyjev.registry import MODELS

from .common import ModelEntry, human_size, make_item, normalize_repo_id
from .workers import DownloadWorker, HealthWorker, RequestWorker

EXAMPLES = {
    "noul": {
        "state": "The user wrote: my order never arrived and I want my money back.",
        "instructions": "Is the user asking for a refund?",
        "criteria": "",
    },
    "choice": {
        "state": "The app crashes when I open settings.",
        "instructions": "Which team should handle this ticket?",
        "criteria": json.dumps(
            {
                "bug": "software defect",
                "billing": "payment issue",
                "question": "how-to question",
            },
            indent=2,
        ),
    },
    "score": {
        "state": "Thanks, that fixed it immediately!",
        "instructions": "How satisfied is the user?",
        "criteria": json.dumps(
            ["very unhappy", "unhappy", "neutral", "happy", "very happy"],
            indent=2,
        ),
    },
}

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TinyJev Server")
        self.resize(1050, 740)

        self.settings = QSettings("TinyJev", "TinyJev-Server-GUI")

        self.model_entries = self.load_model_entries()
        self.model_by_name = {entry.name: entry for entry in self.model_entries}
        self.model_rows = {}
        self.model_status_codes = {}
        self.model_status_text = {}
        self.model_disk_sizes = {}
        self.expected_sizes = {}
        self.expected_files = {}

        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self.read_output)
        self.proc.stateChanged.connect(self.update_state)
        self.proc.finished.connect(self.process_finished)
        self.proc.errorOccurred.connect(self.process_error)

        self.readiness_timer = QTimer(self)
        self.readiness_timer.setInterval(700)
        self.readiness_timer.timeout.connect(self.poll_readiness)

        self.models_timer = QTimer(self)
        self.models_timer.setInterval(2500)
        self.models_timer.timeout.connect(self.refresh_model_table)

        self.server_ready = False
        self.health_worker = None
        self.worker = None
        self.download_worker = None
        self.download_state = None
        self.download_result_received = True

        self.host = QLineEdit(self.settings.value("host", "127.0.0.1"))
        self.host.textChanged.connect(self.update_api_tab_content)
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(int(self.settings.value("port", 8077)))
        self.port.valueChanged.connect(self.update_api_tab_content)

        self.model_combo = QComboBox()
        for entry in self.model_entries:
            self.model_combo.addItem(entry.name)
        saved_model = self.settings.value("model", self.model_entries[0].name)
        model_index = self.model_combo.findText(saved_model)
        if model_index >= 0:
            self.model_combo.setCurrentIndex(model_index)
        self.model_combo.currentTextChanged.connect(self.model_changed)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.toggle_server)
        self.status = QLabel()

        top = QHBoxLayout()
        top.addWidget(QLabel("Host"))
        top.addWidget(self.host)
        top.addWidget(QLabel("Port"))
        top.addWidget(self.port)
        top.addWidget(QLabel("Model"))
        top.addWidget(self.model_combo)
        top.addWidget(self.start_btn)
        top.addWidget(self.status, 1)

        self.log = QPlainTextEdit(readOnly=True)
        self.log.setFont(QFont("Consolas", 9))
        self.log.setMaximumBlockCount(5000)
        log_box = QGroupBox("Server log")
        QVBoxLayout(log_box).addWidget(self.log)

        self.qtype = QComboBox()
        self.qtype.addItems(["noul", "choice", "score"])
        self.qtype.currentTextChanged.connect(self.load_example)

        self.state_in = QPlainTextEdit()
        self.instr_in = QLineEdit()
        self.criteria_in = QPlainTextEdit()
        self.criteria_in.setFont(QFont("Consolas", 9))

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send)

        self.result = QPlainTextEdit(readOnly=True)
        self.result.setFont(QFont("Consolas", 9))

        form = QFormLayout()
        form.addRow("Type", self.qtype)
        form.addRow("State", self.state_in)
        form.addRow("Instructions", self.instr_in)
        form.addRow("Criteria (JSON)", self.criteria_in)
        form.addRow("", self.send_btn)
        form.addRow("Response", self.result)

        play_box = QGroupBox("Playground")
        play_box.setLayout(form)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(log_box)
        split.addWidget(play_box)
        split.setSizes([450, 550])

        server_tab = QWidget()
        server_layout = QVBoxLayout(server_tab)
        server_layout.setContentsMargins(0, 0, 0, 0)
        server_layout.addWidget(split)

        self.models_tab = QWidget()
        self.setup_models_tab()

        self.api_tab = QWidget()
        self.setup_api_tab()

        self.tabs = QTabWidget()
        self.tabs.addTab(server_tab, "Server")
        self.tabs.addTab(self.models_tab, "Models")
        self.tabs.addTab(self.api_tab, "API")

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addLayout(top)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(root)

        self.load_example("noul")
        self.load_model_metadata()
        self.refresh_model_table()
        self.models_timer.start()
        self.update_api_tab_content()
        self.update_state()

    def load_model_entries(self):
        entries = []
        for name, meta in MODELS.items():
            entries.append(
                ModelEntry(
                    name=name,
                    repo_id=meta.get("repo", name),
                    params=str(meta.get("params", "")),
                    description=str(meta.get("what", "")),
                )
            )
        return entries

    def setup_models_tab(self):
        self.models_table = QTableWidget(0, 5)
        self.models_table.setHorizontalHeaderLabels(
            ["Model", "Params", "Size", "Description", "Status"]
        )
        self.models_table.verticalHeader().setVisible(False)
        self.models_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.models_table.setSelectionMode(QTableWidget.SingleSelection)
        self.models_table.setAlternatingRowColors(True)
        header = self.models_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)

        self.download_btn = QPushButton("Download")
        self.download_btn.clicked.connect(self.download_selected)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_download)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self.delete_selected)

        self.open_folder_btn = QPushButton("Open folder")
        self.open_folder_btn.clicked.connect(self.open_selected_folder)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_model_table)

        actions = QHBoxLayout()
        actions.addWidget(self.download_btn)
        actions.addWidget(self.cancel_btn)
        actions.addWidget(self.delete_btn)
        actions.addWidget(self.open_folder_btn)
        actions.addWidget(self.refresh_btn)
        actions.addStretch(1)

        self.download_info = QLabel("No active download")
        self.download_bar = QProgressBar()
        self.download_bar.setMinimum(0)
        self.download_bar.setMaximum(100)
        self.download_bar.setValue(0)

        layout = QVBoxLayout(self.models_tab)
        layout.addWidget(self.models_table)
        layout.addLayout(actions)
        layout.addWidget(self.download_info)
        layout.addWidget(self.download_bar)

    def setup_api_tab(self):
        self.api_base_url = QLineEdit()
        self.api_base_url.setReadOnly(True)

        self.api_curl = QPlainTextEdit(readOnly=True)
        self.api_curl.setFont(QFont("Consolas", 9))
        self.api_curl.setLineWrapMode(QPlainTextEdit.NoWrap)

        self.api_tools = QPlainTextEdit(readOnly=True)
        self.api_tools.setFont(QFont("Consolas", 9))
        self.api_tools.setLineWrapMode(QPlainTextEdit.NoWrap)

        self.api_mcp = QPlainTextEdit(readOnly=True)
        self.api_mcp.setFont(QFont("Consolas", 9))
        self.api_mcp.setLineWrapMode(QPlainTextEdit.NoWrap)

        copy_base_btn = QPushButton("Copy")
        copy_base_btn.clicked.connect(lambda: self.copy_api_text(self.api_base_url.text(), "Base URL copied"))
        copy_curl_btn = QPushButton("Copy")
        copy_curl_btn.clicked.connect(lambda: self.copy_api_text(self.api_curl.toPlainText(), "curl example copied"))
        copy_tools_btn = QPushButton("Copy")
        copy_tools_btn.clicked.connect(
            lambda: self.copy_api_text(self.api_tools.toPlainText(), "OpenAI tools JSON copied")
        )
        copy_mcp_btn = QPushButton("Copy")
        copy_mcp_btn.clicked.connect(lambda: self.copy_api_text(self.api_mcp.toPlainText(), "MCP config copied"))

        self.api_status = QLabel("")

        base_row = QHBoxLayout()
        base_row.addWidget(QLabel("Base URL"))
        base_row.addWidget(self.api_base_url, 1)
        base_row.addWidget(copy_base_btn)

        curl_row = QHBoxLayout()
        curl_row.addWidget(QLabel("curl example"))
        curl_row.addStretch(1)
        curl_row.addWidget(copy_curl_btn)

        tools_row = QHBoxLayout()
        tools_row.addWidget(QLabel("OpenAI tool schema JSON"))
        tools_row.addStretch(1)
        tools_row.addWidget(copy_tools_btn)

        mcp_row = QHBoxLayout()
        mcp_row.addWidget(QLabel("MCP client config (mcpServers)"))
        mcp_row.addStretch(1)
        mcp_row.addWidget(copy_mcp_btn)

        layout = QVBoxLayout(self.api_tab)
        layout.addLayout(base_row)
        layout.addLayout(curl_row)
        layout.addWidget(self.api_curl)
        layout.addLayout(tools_row)
        layout.addWidget(self.api_tools)
        layout.addLayout(mcp_row)
        layout.addWidget(self.api_mcp)
        layout.addWidget(self.api_status)

    def copy_api_text(self, text, message):
        QApplication.clipboard().setText(text)
        self.api_status.setText(message)

    def project_venv_python(self):
        root = Path(__file__).resolve().parents[1]
        return str((root / ".venv" / "Scripts" / "python.exe").resolve())

    def openai_tools_schema(self):
        guidance = (
            "State is the only context. Keep it under about 8K tokens, include all needed facts, "
            "and do not assume memory between calls. Outputs are calibrated probabilities; "
            "low confidence means uncertainty."
        )
        return [
            {
                "type": "function",
                "function": {
                    "name": "jev_yesno",
                    "description": f"Binary yes/no decision. {guidance}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "state": {"type": "string"},
                            "question": {"type": "string"},
                        },
                        "required": ["state", "question"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "jev_choice",
                    "description": f"Pick one option from a labeled set. {guidance}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "state": {"type": "string"},
                            "question": {"type": "string"},
                            "options": {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                            },
                        },
                        "required": ["state", "question", "options"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "jev_score",
                    "description": f"Score across ordered levels from low to high. {guidance}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "state": {"type": "string"},
                            "question": {"type": "string"},
                            "levels": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["state", "question", "levels"],
                    },
                },
            },
        ]

    def update_api_tab_content(self):
        if not hasattr(self, "api_base_url"):
            return

        base_url = self.base_url()
        self.api_base_url.setText(base_url)

        curl_payload = {
            "state": "The customer says: my package never arrived and I want my money back.",
            "questions": {
                "refund": {
                    "type": "noul",
                    "instructions": "Is this a refund request?",
                }
            },
        }
        curl_lines = [
            f"curl -X POST {base_url}/v1/systemone \\",
            "  -H \"Content-Type: application/json\" \\",
            f"  -d '{json.dumps(curl_payload, indent=2)}'",
        ]
        self.api_curl.setPlainText("\n".join(curl_lines))

        self.api_tools.setPlainText(json.dumps(self.openai_tools_schema(), indent=2))

        mcp_config = {
            "mcpServers": {
                "tinyjev": {
                    "command": self.project_venv_python(),
                    "args": ["-m", "tinyjev_gui.mcp"],
                    "env": {"TINYJEV_URL": base_url},
                }
            }
        }
        self.api_mcp.setPlainText(json.dumps(mcp_config, indent=2))

    def load_model_metadata(self):
        api = HfApi()
        for entry in self.model_entries:
            try:
                info = api.model_info(entry.repo_id, files_metadata=True)
                files = [item.rfilename for item in info.siblings if getattr(item, "rfilename", None)]
                sizes = [item.size for item in info.siblings if getattr(item, "size", None)]
                self.expected_files[entry.name] = files
                if sizes:
                    self.expected_sizes[entry.name] = int(sum(sizes))
            except Exception:
                continue

    def model_changed(self, model_name):
        self.settings.setValue("model", model_name)
        self.update_state()

    def base_url(self):
        host = self.host.text().strip() or "127.0.0.1"
        if host == "0.0.0.0":
            host = "127.0.0.1"
        return f"http://{host}:{self.port.value()}"

    def toggle_server(self):
        if self.proc.state() == QProcess.NotRunning:
            self.start_server()
        else:
            self.stop_server()

    def selected_model_name(self):
        return self.model_combo.currentText().strip()

    def start_server(self):
        model_name = self.selected_model_name()
        if not self.model_is_downloaded(model_name):
            answer = QMessageBox.question(
                self,
                "Model not downloaded",
                f"{model_name} is not downloaded yet. Open Models tab to download it now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer == QMessageBox.Yes:
                self.tabs.setCurrentWidget(self.models_tab)
                self.select_model_row(model_name)
            return

        host = self.host.text().strip() or "127.0.0.1"
        port = self.port.value()
        if self.is_port_busy(host, port):
            message = f"Port {port} is already in use on {host}."
            self.log.appendPlainText(message)
            self.status.setText(message)
            return

        self.settings.setValue("host", host)
        self.settings.setValue("port", port)
        self.settings.setValue("model", model_name)

        self.server_ready = False
        self.update_state()

        args = [
            "-u",
            "-m",
            "tinyjev.cli",
            "serve",
            model_name,
            "--host",
            host,
            "--port",
            str(port),
        ]

        self.log.appendPlainText(f"> {sys.executable} {' '.join(args)}")
        self.proc.start(sys.executable, args)
        if not self.proc.waitForStarted(4000):
            self.log.appendPlainText("Failed to start server process.")
            self.update_state()
            return

        self.log.appendPlainText("Waiting for server readiness...")
        self.readiness_timer.start()

    def stop_server(self):
        if self.proc.state() == QProcess.NotRunning:
            return
        self.readiness_timer.stop()
        self.server_ready = False
        self.log.appendPlainText("Stopping server...")
        self.stop_process_tree()
        self.update_state()

    def stop_process_tree(self):
        if self.proc.state() == QProcess.NotRunning:
            return

        pid = int(self.proc.processId())
        self.proc.terminate()
        if self.proc.waitForFinished(2500):
            return

        if sys.platform.startswith("win") and pid > 0:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except Exception as exc:
                self.log.appendPlainText(f"taskkill failed: {exc}")
        else:
            self.proc.kill()

        self.proc.waitForFinished(5000)

    def read_output(self):
        text = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        if text:
            self.log.appendPlainText(text.rstrip())

    def poll_readiness(self):
        if self.proc.state() == QProcess.NotRunning:
            self.readiness_timer.stop()
            return
        if self.health_worker and self.health_worker.isRunning():
            return

        self.health_worker = HealthWorker(self.base_url() + "/health")
        self.health_worker.done.connect(self.handle_health)
        self.health_worker.finished.connect(self.health_done)
        self.health_worker.start()

    def health_done(self):
        self.health_worker = None

    def handle_health(self, ready):
        if ready and not self.server_ready:
            self.server_ready = True
            self.readiness_timer.stop()
            self.log.appendPlainText("Server is ready.")
        elif not ready and self.server_ready:
            self.server_ready = False
            self.readiness_timer.start()
            self.log.appendPlainText("Server readiness check failed.")
        self.update_state()

    def update_state(self, *_):
        running = self.proc.state() != QProcess.NotRunning
        downloading = self.download_worker is not None and self.download_worker.isRunning()

        self.start_btn.setText("Stop" if running else "Start")
        self.host.setEnabled(not running)
        self.port.setEnabled(not running)
        self.model_combo.setEnabled(not running)

        request_running = self.worker is not None and self.worker.isRunning()
        self.send_btn.setEnabled(running and self.server_ready and not request_running)

        self.cancel_btn.setEnabled(downloading)
        self.download_btn.setEnabled(not downloading)
        self.delete_btn.setEnabled(not downloading)

        if running and self.server_ready:
            self.status.setText(f"Running on {self.base_url()}")
        elif running:
            self.status.setText(f"Starting on {self.base_url()}")
        else:
            selected_model = self.selected_model_name()
            if self.model_is_downloaded(selected_model):
                self.status.setText("Stopped")
            else:
                self.status.setText(f"Stopped (download {selected_model} first)")

    def process_finished(self, exit_code, exit_status):
        self.readiness_timer.stop()
        self.server_ready = False
        status_name = "normal" if exit_status == QProcess.NormalExit else "crashed"
        self.log.appendPlainText(f"Server process exited ({status_name}, code {exit_code}).")
        self.update_state()

    def process_error(self, error):
        self.log.appendPlainText(f"Process error: {error}")

    def load_example(self, question_type):
        sample = EXAMPLES[question_type]
        self.state_in.setPlainText(sample["state"])
        self.instr_in.setText(sample["instructions"])
        self.criteria_in.setPlainText(sample["criteria"])
        self.criteria_in.setEnabled(question_type != "noul")

    def send(self):
        if self.proc.state() == QProcess.NotRunning or not self.server_ready:
            self.result.setPlainText("Server is not ready.")
            self.update_state()
            return

        question = {
            "type": self.qtype.currentText(),
            "instructions": self.instr_in.text(),
        }
        if question["type"] != "noul":
            try:
                question["criteria"] = json.loads(self.criteria_in.toPlainText())
            except ValueError as exc:
                self.result.setPlainText(f"Criteria is not valid JSON: {exc}")
                return

        payload = {"state": self.state_in.toPlainText(), "questions": {"result": question}}

        self.result.setPlainText("Waiting...")
        self.worker = RequestWorker(self.base_url() + "/v1/systemone", payload)
        self.worker.done.connect(self.show_result)
        self.worker.finished.connect(self.request_done)
        self.worker.start()
        self.update_state()

    def request_done(self):
        self.worker = None
        self.update_state()

    def show_result(self, text):
        self.result.setPlainText(text)

    def is_port_busy(self, host, port):
        probe_host = host if host and host != "0.0.0.0" else "127.0.0.1"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((probe_host, int(port)))
            except OSError:
                return True
        return False

    def model_is_downloaded(self, model_name):
        status = self.model_status_codes.get(model_name)
        if status is None:
            self.refresh_model_table()
            status = self.model_status_codes.get(model_name)
        return status == "downloaded"

    def find_cached_repo(self, repo_id, repo_map=None):
        target = normalize_repo_id(repo_id)
        repo = None
        if repo_map is not None:
            repo = repo_map.get(target)
        else:
            try:
                cache = scan_cache_dir()
                repo_map = {
                    normalize_repo_id(item.repo_id): item
                    for item in cache.repos
                    if item.repo_type == "model"
                }
                repo = repo_map.get(target)
            except Exception:
                repo = None

        if repo is not None:
            return repo, Path(repo.repo_path)

        owner, name = repo_id.split("/", 1)
        expected_name = f"models--{owner}--{name}"
        cache_root = Path(HF_HUB_CACHE)
        for child in cache_root.glob("models--*--*"):
            if child.name.lower() == expected_name.lower():
                return None, child

        return None, cache_root / expected_name

    def folder_size(self, path):
        if path is None or not path.exists():
            return 0
        if path.is_file():
            try:
                return path.stat().st_size
            except OSError:
                return 0
        total = 0
        for child in path.rglob("*"):
            if child.is_file():
                try:
                    total += child.stat().st_size
                except OSError:
                    continue
        return total

    def snapshot_file_names(self, repo_info):
        if not repo_info or not repo_info.revisions:
            return set()
        main_ref = repo_info.refs.get("main")
        revision = main_ref
        if revision is None:
            revision = max(repo_info.revisions, key=lambda item: item.last_modified)
        return {file.file_name for file in revision.files}

    def refresh_download_state_from_cache(self, repo_map):
        if not self.download_state:
            return

        model_name = self.download_state.get("model")
        entry = self.model_by_name.get(model_name)
        if entry is None:
            return

        _, folder = self.find_cached_repo(entry.repo_id, repo_map)
        current_size = self.folder_size(folder)
        previous_size = int(self.download_state.get("downloaded", 0))
        now = time.time()
        previous_time = float(self.download_state.get("updated_at", now))

        if current_size > previous_size:
            delta_t = max(now - previous_time, 1e-3)
            self.download_state["speed"] = (current_size - previous_size) / delta_t
            self.download_state["downloaded"] = current_size

        self.download_state["updated_at"] = now

    def revision_note(self, repo_info):
        revisions = getattr(repo_info, "revisions", None) or []
        revision_count = len(revisions)
        if revision_count > 1:
            return f" • {revision_count} revisions cached"
        return ""

    def model_status_for_entry(self, entry, repo_map):
        if self.download_state and self.download_state.get("model") == entry.name:
            downloaded = self.download_state.get("downloaded", 0)
            total = self.download_state.get("total", 0)
            speed = self.download_state.get("speed", 0.0)
            percent = 0.0
            if total > 0:
                percent = max(0.0, min(100.0, downloaded * 100.0 / total))
            speed_part = f" • {human_size(speed)}/s" if speed > 0 else ""
            text = (
                f"Downloading {percent:.1f}% "
                f"({human_size(downloaded)} / {human_size(total)}){speed_part}"
            )
            return "downloading", text, downloaded

        repo_info, folder = self.find_cached_repo(entry.repo_id, repo_map)

        if repo_info is not None:
            repo_size = int(repo_info.size_on_disk or 0)
            revision_note = self.revision_note(repo_info)

            expected_files = set(self.expected_files.get(entry.name, []))
            local_files = self.snapshot_file_names(repo_info)
            if expected_files:
                downloaded = expected_files.issubset(local_files)
            else:
                downloaded = bool(local_files)

            if downloaded:
                return "downloaded", f"Downloaded ({human_size(repo_size)}){revision_note}", repo_size
            if repo_size > 0:
                return "partial", f"Partial ({human_size(repo_size)}){revision_note}", repo_size
            return "missing", "Not downloaded", 0

        raw_size = self.folder_size(folder)
        if raw_size > 0:
            return "partial", f"Partial ({human_size(raw_size)})", raw_size
        return "missing", "Not downloaded", 0

    def refresh_model_table(self):
        try:
            cache = scan_cache_dir()
            repo_map = {
                normalize_repo_id(item.repo_id): item
                for item in cache.repos
                if item.repo_type == "model"
            }
        except Exception:
            repo_map = {}

        self.refresh_download_state_from_cache(repo_map)

        selected_name = self.current_selected_table_model()
        self.models_table.setRowCount(len(self.model_entries))
        self.model_rows.clear()

        for row, entry in enumerate(self.model_entries):
            self.model_rows[entry.name] = row
            size_value = self.expected_sizes.get(entry.name)
            size_text = human_size(size_value) if size_value else "Unknown"

            status_code, status_text, disk_size = self.model_status_for_entry(entry, repo_map)
            self.model_status_codes[entry.name] = status_code
            self.model_status_text[entry.name] = status_text
            self.model_disk_sizes[entry.name] = disk_size

            self.models_table.setItem(row, 0, make_item(entry.name))
            self.models_table.setItem(row, 1, make_item(entry.params or "-"))
            self.models_table.setItem(row, 2, make_item(size_text))
            self.models_table.setItem(row, 3, make_item(entry.description or "-"))
            self.models_table.setItem(row, 4, make_item(status_text))

        if selected_name:
            self.select_model_row(selected_name)
        elif self.models_table.rowCount() > 0:
            self.models_table.selectRow(0)

        if self.download_state and self.download_state.get("total", 0) > 0:
            total = self.download_state["total"]
            current = self.download_state.get("downloaded", 0)
            percent = max(0, min(100, int(current * 100 / total)))
            self.download_bar.setValue(percent)
        elif self.download_worker is None:
            self.download_bar.setValue(0)
            self.download_info.setText("No active download")

        self.update_state()

    def current_selected_table_model(self):
        indexes = self.models_table.selectionModel().selectedRows()
        if not indexes:
            return None
        row = indexes[0].row()
        item = self.models_table.item(row, 0)
        return item.text() if item else None

    def select_model_row(self, model_name):
        row = self.model_rows.get(model_name)
        if row is None:
            return
        self.models_table.selectRow(row)

    def download_selected(self):
        if self.download_worker and self.download_worker.isRunning():
            return

        model_name = self.current_selected_table_model() or self.selected_model_name()
        entry = self.model_by_name.get(model_name)
        if entry is None:
            return

        code = self.model_status_codes.get(model_name)
        if code == "downloaded":
            QMessageBox.information(self, "Already downloaded", f"{model_name} is already downloaded.")
            return

        initial_bytes = int(self.model_disk_sizes.get(model_name, 0))
        total_hint = int(self.expected_sizes.get(model_name, 0))

        self.download_worker = DownloadWorker(
            model_name=model_name,
            repo_id=entry.repo_id,
            initial_bytes=initial_bytes,
            total_hint=total_hint,
        )
        self.download_worker.started_model.connect(self.download_started)
        self.download_worker.progress.connect(self.download_progress)
        self.download_worker.finished_model.connect(self.download_finished)
        self.download_worker.finished.connect(self.download_thread_done)

        self.download_result_received = False
        self.download_state = {
            "model": model_name,
            "downloaded": initial_bytes,
            "total": total_hint,
            "speed": 0.0,
            "updated_at": time.time(),
        }

        self.download_info.setText(f"Downloading {model_name}...")
        self.download_bar.setValue(0)
        self.download_worker.start()
        self.refresh_model_table()

    def download_started(self, model_name, downloaded, total):
        downloaded = int(downloaded or 0)
        total = int(total or 0)
        self.download_state = {
            "model": model_name,
            "downloaded": downloaded,
            "total": total,
            "speed": 0.0,
            "updated_at": time.time(),
        }
        self.refresh_model_table()

    def download_progress(self, model_name, downloaded, total, speed):
        downloaded = int(downloaded or 0)
        total = int(total or 0)
        self.download_state = {
            "model": model_name,
            "downloaded": downloaded,
            "total": total,
            "speed": speed,
            "updated_at": time.time(),
        }
        if total > 0:
            percent = max(0, min(100, int(downloaded * 100 / total)))
            self.download_bar.setValue(percent)
        speed_part = f" • {human_size(speed)}/s" if speed > 0 else ""
        if total > 0:
            text = (
                f"Downloading {model_name}: {human_size(downloaded)} / "
                f"{human_size(total)}{speed_part}"
            )
        else:
            text = f"Downloading {model_name}: {human_size(downloaded)}{speed_part}"
        self.download_info.setText(text)
        self.refresh_model_table()

    def download_finished(self, model_name, cancelled, error_text):
        self.download_result_received = True
        if cancelled:
            self.download_info.setText(f"Download canceled for {model_name}.")
        elif error_text:
            self.download_info.setText(f"Download failed for {model_name}: {error_text}")
            QMessageBox.warning(self, "Download failed", f"{model_name}: {error_text}")
        else:
            self.download_info.setText(f"Download complete for {model_name}.")

        self.download_state = None
        self.download_bar.setValue(0)
        self.refresh_model_table()

    def download_thread_done(self):
        if not self.download_result_received:
            self.download_state = None
            self.download_bar.setValue(0)
            self.download_info.setText("Download stopped.")
        self.download_worker = None
        self.refresh_model_table()
        self.update_state()

    def cancel_download(self):
        if self.download_worker and self.download_worker.isRunning():
            self.download_worker.cancel()
            self.download_info.setText("Stopping download...")
            QTimer.singleShot(3000, self.force_cancel_download)

    def force_cancel_download(self):
        worker = self.download_worker
        if not worker or not worker.isRunning():
            return
        worker.terminate()
        worker.wait(2000)
        if self.download_worker is worker:
            self.download_worker = None
        self.download_state = None
        self.download_result_received = True
        self.download_info.setText("Download canceled.")
        self.download_bar.setValue(0)
        self.refresh_model_table()

    def delete_selected(self):
        model_name = self.current_selected_table_model()
        if not model_name:
            return
        entry = self.model_by_name.get(model_name)
        if entry is None:
            return

        if self.download_worker and self.download_worker.isRunning():
            QMessageBox.warning(self, "Download active", "Cancel the active download before deleting cache.")
            return

        answer = QMessageBox.question(
            self,
            "Delete model cache",
            f"Delete cached files for {model_name}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        freed = self.delete_model_cache(entry.repo_id)
        self.download_info.setText(f"Deleted cache for {model_name} ({human_size(freed)} freed).")
        self.refresh_model_table()

    def delete_model_cache(self, repo_id):
        try:
            cache = scan_cache_dir()
            repo_map = {
                normalize_repo_id(item.repo_id): item
                for item in cache.repos
                if item.repo_type == "model"
            }
        except Exception:
            repo_map = {}

        _, folder = self.find_cached_repo(repo_id, repo_map)
        before = self.folder_size(folder)

        if folder and folder.exists():
            for _ in range(8):
                try:
                    shutil.rmtree(folder)
                    break
                except PermissionError:
                    time.sleep(1)
                except OSError:
                    break

        after = self.folder_size(folder)
        return max(0, before - after)

    def open_selected_folder(self):
        model_name = self.current_selected_table_model()
        if not model_name:
            return
        entry = self.model_by_name.get(model_name)
        if entry is None:
            return

        repo_info, folder = self.find_cached_repo(entry.repo_id)
        target = Path(repo_info.repo_path) if repo_info is not None else folder

        if not target.exists():
            QMessageBox.information(self, "Folder missing", "No local cache folder exists for this model.")
            return

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def closeEvent(self, event):
        self.models_timer.stop()

        if self.download_worker and self.download_worker.isRunning():
            self.download_worker.cancel()
            self.download_worker.wait(6000)

        self.stop_server()
        super().closeEvent(event)



def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
