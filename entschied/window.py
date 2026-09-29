import errno
import getpass
import json
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import scan_cache_dir, try_to_load_from_cache
from huggingface_hub.constants import HF_HUB_CACHE
from PySide6.QtCore import QObject, QProcess, QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QSystemTrayIcon,
    QTabWidget,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)
from tinyjev.registry import MODELS

from .vulkan_setup import EXTRA_MODELS, gguf_expected_size, is_jevk5
from .common import ModelEntry, human_size, make_item, normalize_repo_id
from .runtime import directml_available, executable_path, is_frozen_app, torch_available

try:
    from .gpu_setup import default_runtime_dir, runtime_python
except Exception:
    def default_runtime_dir() -> Path:
        base = Path.home()
        return base / "AppData" / "Local" / "Entschied" / "gpu-runtime"

    def runtime_python(runtime_dir: Path) -> Path:
        return runtime_dir / "Scripts" / "python.exe"

from .vulkan_setup import (
    DIRECTML_SIZE_LIMIT_BYTES,
    cached_gguf_bundle,
    default_gguf_quant,
    directml_allowed,
    find_llama_server,
    gguf_allow_patterns,
    gguf_quants,
    gguf_repo_id,
    gguf_status,
    llama_has_vulkan,
)
from .workers import (
    DownloadWorker,
    GgufConvertWorker,
    GpuRuntimeSetupWorker,
    HealthWorker,
    ModelMetadataWorker,
    RequestWorker,
)
from .runtime import NO_WINDOW

DEFAULT_SAFETY_THRESHOLD = 0.70

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

class SingleInstanceBridge(QObject):
    def __init__(self, name, message_handler, parent=None):
        super().__init__(parent)
        self.name = name
        self.message_handler = message_handler
        self.server = QLocalServer(self)
        self.server.newConnection.connect(self.handle_new_connection)

    def listen(self):
        if self.server.listen(self.name):
            return True

        probe = QLocalSocket()
        probe.connectToServer(self.name)
        if probe.waitForConnected(250):
            probe.disconnectFromServer()
            return False

        QLocalServer.removeServer(self.name)
        return self.server.listen(self.name)

    @staticmethod
    def send_message(name, message):
        socket_client = QLocalSocket()
        socket_client.connectToServer(name)
        if not socket_client.waitForConnected(250):
            return False
        payload = (message + "\n").encode("utf-8")
        socket_client.write(payload)
        socket_client.flush()
        socket_client.waitForBytesWritten(250)
        socket_client.disconnectFromServer()
        return True

    def handle_new_connection(self):
        while self.server.hasPendingConnections():
            socket_client = self.server.nextPendingConnection()
            if socket_client is None:
                continue
            socket_client.waitForReadyRead(250)
            payload = bytes(socket_client.readAll()).decode("utf-8", errors="replace").strip()
            socket_client.disconnectFromServer()
            if payload:
                self.message_handler(payload)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Entschied")
        self.resize(1050, 740)

        self.settings = QSettings("Entschied", "Entschied")

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
        self.has_seen_ready = False
        self.unexpected_stop_message = ""
        self.start_error_message = ""
        self.health_worker = None
        self.metadata_worker = None
        self.worker = None
        self.download_worker = None
        self.download_state = None
        self.download_result_received = True
        self.is_quitting = False
        self.gpu_setup_worker = None
        self.convert_worker = None
        self.last_start_device = None
        self.stop_requested = False
        self.running_model_name = None
        self.safety_hint_visible = False

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

        self.device_combo = QComboBox()
        self.configure_device_options()
        self.set_selected_device(self.initial_device_key())
        self.device_combo.currentIndexChanged.connect(self.device_changed)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.toggle_server)
        self.status = QLabel()

        self.start_on_open_checkbox = QCheckBox("Start server when app opens")
        self.start_on_open_checkbox.setChecked(self.setting_bool("start_on_open", False))
        self.start_on_open_checkbox.toggled.connect(self.start_on_open_changed)

        self.start_on_login_checkbox = QCheckBox("Start app on login")
        self.start_on_login_checkbox.setEnabled(sys.platform.startswith("win"))
        if sys.platform.startswith("win"):
            self.start_on_login_checkbox.setChecked(self.start_on_login_enabled())
            self.start_on_login_checkbox.toggled.connect(self.start_on_login_changed)

        saved_threshold = float(self.settings.value("safety_threshold", DEFAULT_SAFETY_THRESHOLD))
        saved_threshold = max(0.50, min(0.99, saved_threshold))
        self.safety_threshold_enabled = QCheckBox("Safety threshold")
        self.safety_threshold_enabled.setToolTip(
            "Server default for every client. Answers below this confidence come back as \"unsure\"\n"
            "with the raw pick included. Applied when the server starts; a request can override it\n"
            "with min_confidence."
        )
        self.safety_threshold_enabled.setChecked(self.setting_bool("safety_threshold_enabled", True))
        self.safety_threshold_enabled.toggled.connect(self.safety_threshold_toggled)

        self.safety_threshold_slider = QSlider(Qt.Horizontal)
        self.safety_threshold_slider.setRange(50, 99)
        self.safety_threshold_slider.setSingleStep(1)
        self.safety_threshold_slider.setPageStep(1)
        self.safety_threshold_slider.setValue(int(round(saved_threshold * 100)))
        self.safety_threshold_slider.valueChanged.connect(self.safety_threshold_changed)

        self.safety_threshold_value = QLabel()
        self.update_safety_threshold_label(saved_threshold)
        self.safety_threshold_slider.setEnabled(self.safety_threshold_enabled.isChecked())

        top = QHBoxLayout()
        top.addWidget(QLabel("Host"))
        top.addWidget(self.host)
        top.addWidget(QLabel("Port"))
        top.addWidget(self.port)
        top.addWidget(QLabel("Model"))
        top.addWidget(self.model_combo)
        top.addWidget(QLabel("Device"))
        top.addWidget(self.device_combo)
        top.addWidget(self.start_btn)
        top.addWidget(self.status, 1)

        options = QHBoxLayout()
        options.addWidget(self.start_on_open_checkbox)
        options.addWidget(self.start_on_login_checkbox)
        options.addSpacing(12)
        options.addWidget(self.safety_threshold_enabled)
        options.addWidget(self.safety_threshold_slider, 1)
        options.addWidget(self.safety_threshold_value)
        options.addStretch(1)

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

        self.playground_cutoff_enabled = QCheckBox("Override for this request")
        self.playground_cutoff_enabled.setToolTip(
            "Off: Playground requests use the Safety threshold from the top bar.\n"
            "On: sends min_confidence with the request, like an API client can. Other clients are not affected."
        )
        self.playground_cutoff_enabled.setChecked(self.setting_bool("playground_cutoff_enabled", False))
        self.playground_cutoff_enabled.toggled.connect(self.playground_cutoff_toggled)

        self.playground_cutoff = QDoubleSpinBox()
        self.playground_cutoff.setRange(0.0, 1.0)
        self.playground_cutoff.setDecimals(2)
        self.playground_cutoff.setSingleStep(0.01)
        self.playground_cutoff.setValue(float(self.settings.value("playground_min_confidence", DEFAULT_SAFETY_THRESHOLD)))
        self.playground_cutoff.valueChanged.connect(self.playground_cutoff_changed)
        self.playground_cutoff.setEnabled(self.playground_cutoff_enabled.isChecked())

        cutoff_row = QHBoxLayout()
        cutoff_row.setContentsMargins(0, 0, 0, 0)
        cutoff_row.addWidget(self.playground_cutoff_enabled)
        cutoff_row.addWidget(QLabel("Min confidence"))
        cutoff_row.addWidget(self.playground_cutoff)
        cutoff_row.addStretch(1)

        self.cutoff_note = QLabel()
        self.cutoff_note.setWordWrap(True)
        self.cutoff_note.setStyleSheet("color: #8a9a90;")

        cutoff_box = QVBoxLayout()
        cutoff_box.setContentsMargins(0, 0, 0, 0)
        cutoff_box.addLayout(cutoff_row)
        cutoff_box.addWidget(self.cutoff_note)

        cutoff_wrap = QWidget()
        cutoff_wrap.setLayout(cutoff_box)

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send)

        self.result = QPlainTextEdit(readOnly=True)
        self.result.setFont(QFont("Consolas", 9))

        form = QFormLayout()
        form.addRow("Type", self.qtype)
        form.addRow("State", self.state_in)
        form.addRow("Instructions", self.instr_in)
        form.addRow("Criteria (JSON)", self.criteria_in)
        form.addRow("Cutoff", cutoff_wrap)
        self.update_cutoff_note()
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
        layout.addLayout(options)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(root)

        self.load_example("noul")
        self.start_model_metadata_load()
        self.refresh_model_table()
        self.models_timer.start()
        self.setup_tray()
        self.update_api_tab_content()
        self.update_state()

        if self.start_on_open_checkbox.isChecked():
            QTimer.singleShot(0, self.start_server)

    def load_model_entries(self):
        entries = []
        for name, meta in {**MODELS, **EXTRA_MODELS}.items():
            entries.append(
                ModelEntry(
                    name=name,
                    repo_id=meta.get("repo", name),
                    params=str(meta.get("params", "")),
                    description=str(meta.get("what", "")),
                )
            )
        return entries

    def setting_bool(self, key, default=False):
        raw = self.settings.value(key, default)
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def gpu_runtime_dir(self):
        raw = self.settings.value("gpu_runtime_dir", str(default_runtime_dir()))
        return Path(str(raw))

    def gpu_runtime_python(self):
        custom = str(self.settings.value("gpu_python_path", "")).strip()
        if custom:
            return Path(custom)
        return runtime_python(self.gpu_runtime_dir())

    def gpu_runtime_exists(self):
        return self.gpu_runtime_python().exists()

    def vulkan_runtime_available(self):
        try:
            server = find_llama_server(auto_download=False)
        except Exception:
            return False
        return llama_has_vulkan(str(server))

    def legacy_torch_modes_enabled(self):
        return not is_frozen_app() and torch_available()

    def configure_device_options(self):
        self.device_combo.clear()
        self.device_combo.addItem("CPU (llama.cpp)", "cpu")
        self.device_combo.addItem("GPU (Vulkan)", "vulkan")
        if self.legacy_torch_modes_enabled():
            self.device_combo.addItem("CPU (PyTorch)", "cpu_torch")
            if directml_available() or self.gpu_runtime_exists():
                self.device_combo.addItem("GPU (DirectML)", "gpu")

    def normalize_device_value(self, raw_value):
        raw = str(raw_value or "").strip().lower()
        aliases = {
            "gpu": "gpu",
            "cpu": "cpu",
            "vulkan": "vulkan",
            "cpu_torch": "cpu_torch",
            "cpu (llama.cpp)": "cpu",
            "gpu (directml)": "gpu",
            "gpu (vulkan)": "vulkan",
            "cpu (pytorch)": "cpu_torch",
        }
        if raw in aliases:
            return aliases[raw]
        upper = str(raw_value or "").strip().upper()
        if upper in {"CPU", "GPU", "VULKAN"}:
            return upper.lower()
        return ""

    def initial_device_key(self):
        saved = self.normalize_device_value(self.settings.value("device", ""))
        if saved:
            if saved in {"cpu_torch", "gpu"} and not self.legacy_torch_modes_enabled():
                return "cpu"
            return saved
        if self.vulkan_runtime_available():
            return "vulkan"
        return "cpu"

    def set_selected_device(self, device_key):
        wanted = self.normalize_device_value(device_key)
        if not wanted:
            wanted = "cpu"
        for idx in range(self.device_combo.count()):
            if self.device_combo.itemData(idx) == wanted:
                self.device_combo.setCurrentIndex(idx)
                return
        self.device_combo.setCurrentIndex(0)

    def selected_device(self):
        value = self.device_combo.currentData()
        return self.normalize_device_value(value) or "cpu"

    def uses_gguf_downloads(self):
        return self.selected_device() in {"cpu", "vulkan"}

    def device_changed(self, *_):
        self.settings.setValue("device", self.selected_device())
        self.update_gguf_quant_selector()
        self.refresh_model_table()

    def gguf_quant_key(self, model_name):
        return f"gguf_quant/{model_name}"

    def selected_gguf_quant(self, model_name):
        quants = gguf_quants(model_name)
        if not quants:
            return default_gguf_quant(model_name)
        saved = str(self.settings.value(self.gguf_quant_key(model_name), "")).strip()
        if saved in quants:
            return saved
        preferred = default_gguf_quant(model_name)
        if preferred in quants:
            return preferred
        return quants[0]

    def update_gguf_quant_selector(self):
        if not hasattr(self, "gguf_quant_combo"):
            return
        model_name = self.current_selected_table_model() or self.selected_model_name()
        quants = gguf_quants(model_name)
        selected = self.selected_gguf_quant(model_name)
        self.gguf_quant_combo.blockSignals(True)
        self.gguf_quant_combo.clear()
        for quant in quants:
            self.gguf_quant_combo.addItem(quant)
        idx = self.gguf_quant_combo.findText(selected)
        if idx >= 0:
            self.gguf_quant_combo.setCurrentIndex(idx)
        self.gguf_quant_combo.blockSignals(False)
        running = self.proc.state() != QProcess.NotRunning
        show_quant = self.uses_gguf_downloads()
        self.gguf_quant_label.setVisible(show_quant)
        self.gguf_quant_combo.setVisible(show_quant)
        enabled = show_quant and len(quants) > 1 and self.download_worker is None and not running
        self.gguf_quant_combo.setEnabled(enabled)
        self.gguf_quant_label.setEnabled(show_quant and len(quants) > 1 and not running)

    def gguf_quant_changed(self, quant):
        model_name = self.current_selected_table_model() or self.selected_model_name()
        if not model_name or not quant:
            return
        self.settings.setValue(self.gguf_quant_key(model_name), quant)
        self.refresh_model_table()

    def model_table_selection_changed(self):
        self.update_gguf_quant_selector()

    def start_on_open_changed(self, checked):
        self.settings.setValue("start_on_open", bool(checked))

    def startup_command(self):
        if is_frozen_app():
            return f'"{executable_path()}" --minimized'

        pythonw_path = Path(self.project_venv_python()).with_name("pythonw.exe")
        if not pythonw_path.exists():
            pythonw_path = Path(self.project_venv_python())
        return f'"{pythonw_path}" -m entschied --minimized'

    def startup_command_executable(self, command):
        raw = str(command or "").strip()
        if not raw:
            return None
        try:
            parts = shlex.split(raw, posix=False)
        except ValueError:
            return None
        if not parts:
            return None
        return Path(parts[0].strip('"'))

    def startup_targets(self):
        targets = []
        if is_frozen_app():
            try:
                targets.append(str(executable_path().resolve()))
            except OSError:
                pass
            return set(targets)

        python_path = Path(self.project_venv_python())
        pythonw_path = python_path.with_name("pythonw.exe")
        for candidate in (python_path, pythonw_path):
            if candidate.exists():
                try:
                    targets.append(str(candidate.resolve()))
                except OSError:
                    continue
        return set(targets)

    def startup_command_matches_current_repo(self, command):
        executable = self.startup_command_executable(command)
        if executable is None or not executable.exists():
            return False
        try:
            resolved_executable = str(executable.resolve())
        except OSError:
            return False
        if resolved_executable not in self.startup_targets():
            return False
        if is_frozen_app():
            return "--minimized" in str(command)
        return "-m entschied" in str(command)

    def start_on_login_enabled(self):
        if not sys.platform.startswith("win"):
            return False
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\\Microsoft\\Windows\\CurrentVersion\\Run") as key:
                value, _ = winreg.QueryValueEx(key, "Entschied")
        except OSError:
            return False

        command = str(value).strip()
        if not command:
            return False
        if self.startup_command_matches_current_repo(command):
            return True

        ok, _ = self.set_start_on_login(True)
        return ok

    def set_start_on_login(self, enabled):
        if not sys.platform.startswith("win"):
            return False, "Start on login is only supported on Windows."

        try:
            import winreg

            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, r"Software\\Microsoft\\Windows\\CurrentVersion\\Run") as key:
                if enabled:
                    winreg.SetValueEx(key, "Entschied", 0, winreg.REG_SZ, self.startup_command())
                else:
                    try:
                        winreg.DeleteValue(key, "Entschied")
                    except FileNotFoundError:
                        pass
            return True, ""
        except OSError as exc:
            return False, str(exc)

    def start_on_login_changed(self, checked):
        ok, error_text = self.set_start_on_login(bool(checked))
        if ok:
            return
        QMessageBox.warning(self, "Start on login", f"Failed to update login startup: {error_text}")
        self.start_on_login_checkbox.blockSignals(True)
        self.start_on_login_checkbox.setChecked(not checked)
        self.start_on_login_checkbox.blockSignals(False)

    def setup_tray(self):
        tray_icon = self.windowIcon()
        if tray_icon.isNull():
            tray_icon = self.style().standardIcon(QStyle.SP_ComputerIcon)

        self.tray_menu = QMenu(self)
        self.tray_show_action = self.tray_menu.addAction("Show")
        self.tray_show_action.triggered.connect(self.show_from_tray)
        self.tray_toggle_action = self.tray_menu.addAction("Start server")
        self.tray_toggle_action.triggered.connect(self.toggle_server)
        self.tray_menu.addSeparator()
        self.tray_quit_action = self.tray_menu.addAction("Quit")
        self.tray_quit_action.triggered.connect(self.quit_from_tray)

        self.tray_icon = QSystemTrayIcon(tray_icon, self)
        self.tray_icon.setToolTip("Entschied")
        self.tray_icon.setContextMenu(self.tray_menu)
        self.tray_icon.activated.connect(self.tray_activated)
        self.tray_icon.show()

    def tray_activated(self, reason):
        if reason in (QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick):
            self.show_from_tray()

    def show_from_tray(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def handle_single_instance_message(self, payload):
        if payload == "show":
            self.show_from_tray()

    def active_work_labels(self):
        labels = []
        if self.download_worker and self.download_worker.isRunning():
            labels.append("download")
        if self.gpu_setup_worker and self.gpu_setup_worker.isRunning():
            labels.append("DirectML runtime setup")
        if self.convert_worker and self.convert_worker.isRunning():
            labels.append("GGUF conversion")
        if self.proc.state() != QProcess.NotRunning:
            labels.append("server")
        return labels

    def confirm_quit_if_busy(self):
        labels = self.active_work_labels()
        if not labels:
            return True
        text = ", ".join(labels)
        answer = QMessageBox.question(
            self,
            "Quit Entschied",
            f"{text} is still running. Quit anyway?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def cleanup_for_exit(self):
        self.models_timer.stop()

        if self.metadata_worker and self.metadata_worker.isRunning():
            self.metadata_worker.wait(500)

        if self.download_worker and self.download_worker.isRunning():
            self.download_worker.cancel()
            self.download_info.setText("Cancelling download...")
            self.download_worker.wait(15000)

        if self.gpu_setup_worker and self.gpu_setup_worker.isRunning():
            self.gpu_setup_worker.cancel()
            self.gpu_setup_worker.wait(15000)

        if self.convert_worker and self.convert_worker.isRunning():
            self.convert_worker.cancel()
            self.convert_worker.wait(15000)

        self.stop_server()
        if hasattr(self, "tray_icon"):
            self.tray_icon.hide()

    def quit_from_tray(self):
        if not self.confirm_quit_if_busy():
            return
        self.is_quitting = True
        self.close()

    def setup_models_tab(self):
        self.models_table = QTableWidget(0, 6)
        self.models_table.setHorizontalHeaderLabels(
            ["Model", "Params", "Size", "Description", "Status", "GGUF"]
        )
        self.models_table.verticalHeader().setVisible(False)
        self.models_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.models_table.setSelectionMode(QTableWidget.SingleSelection)
        self.models_table.setAlternatingRowColors(True)
        self.models_table.itemSelectionChanged.connect(self.model_table_selection_changed)
        header = self.models_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)

        self.download_btn = QPushButton("Download")
        self.download_btn.clicked.connect(self.download_selected)

        self.gguf_quant_label = QLabel("GGUF quant")
        self.gguf_quant_combo = QComboBox()
        self.gguf_quant_combo.currentTextChanged.connect(self.gguf_quant_changed)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_download)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self.delete_selected)

        self.open_folder_btn = QPushButton("Open folder")
        self.open_folder_btn.clicked.connect(self.open_selected_folder)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_model_table)

        self.convert_btn = QPushButton("Convert GGUF")
        self.convert_btn.clicked.connect(self.convert_selected_model)

        self.setup_gpu_btn = QPushButton("Setup DirectML runtime")
        self.setup_gpu_btn.clicked.connect(self.setup_gpu_runtime)

        if is_frozen_app():
            self.convert_btn.setVisible(False)
            self.setup_gpu_btn.setVisible(False)
        elif not self.legacy_torch_modes_enabled():
            self.setup_gpu_btn.setVisible(False)

        actions = QHBoxLayout()
        actions.addWidget(self.download_btn)
        actions.addWidget(self.gguf_quant_label)
        actions.addWidget(self.gguf_quant_combo)
        actions.addWidget(self.cancel_btn)
        actions.addWidget(self.delete_btn)
        actions.addWidget(self.open_folder_btn)
        actions.addWidget(self.refresh_btn)
        actions.addWidget(self.convert_btn)
        actions.addWidget(self.setup_gpu_btn)
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
        self.update_gguf_quant_selector()

    def setup_gpu_runtime(self):
        if self.gpu_setup_worker and self.gpu_setup_worker.isRunning():
            return

        runtime_dir = self.gpu_runtime_dir()
        repo_root = Path(__file__).resolve().parents[1]

        self.setup_gpu_btn.setEnabled(False)
        self.download_info.setText(f"Preparing DirectML runtime in {runtime_dir}...")
        self.gpu_setup_worker = GpuRuntimeSetupWorker(runtime_dir=runtime_dir, repo_root=repo_root)
        self.gpu_setup_worker.log.connect(self.log.appendPlainText)
        self.gpu_setup_worker.done.connect(self.gpu_runtime_setup_done)
        self.gpu_setup_worker.finished.connect(self.gpu_runtime_setup_finished)
        self.gpu_setup_worker.start()

    def gpu_runtime_setup_done(self, ok, message):
        if ok:
            self.download_info.setText(f"DirectML runtime ready: {message}")
            self.log.appendPlainText(f"DirectML runtime ready: {message}")
            return

        if str(message).strip().lower() == "cancelled":
            self.download_info.setText("DirectML runtime setup cancelled")
            self.log.appendPlainText("DirectML runtime setup cancelled")
            return

        self.download_info.setText("DirectML runtime setup failed")
        self.log.appendPlainText(f"DirectML runtime setup failed: {message}")
        QMessageBox.warning(self, "DirectML runtime", f"DirectML runtime setup failed: {message}")

    def gpu_runtime_setup_finished(self):
        self.gpu_setup_worker = None
        self.setup_gpu_btn.setEnabled(True)

    def convert_selected_model(self):
        if self.convert_worker and self.convert_worker.isRunning():
            return

        model_name = self.current_selected_table_model() or self.selected_model_name()
        if not model_name:
            return

        self.convert_btn.setEnabled(False)
        self.download_info.setText(f"Converting {model_name} to GGUF...")
        self.convert_worker = GgufConvertWorker(model_name)
        self.convert_worker.log.connect(self.log.appendPlainText)
        self.convert_worker.done.connect(self.convert_finished)
        self.convert_worker.finished.connect(self.convert_thread_finished)
        self.convert_worker.start()

    def convert_finished(self, ok, message):
        if ok:
            self.download_info.setText("GGUF conversion complete")
            self.log.appendPlainText(f"GGUF conversion complete: {message}")
            return

        if str(message).strip().lower() == "cancelled":
            self.download_info.setText("GGUF conversion cancelled")
            self.log.appendPlainText("GGUF conversion cancelled")
            return

        self.download_info.setText("GGUF conversion failed")
        self.log.appendPlainText(f"GGUF conversion failed: {message}")
        QMessageBox.warning(self, "GGUF conversion", f"Conversion failed: {message}")

    def convert_thread_finished(self):
        self.convert_worker = None
        self.convert_btn.setEnabled(True)
        self.refresh_model_table()

    def setup_api_tab(self):
        self.api_base_url = QLineEdit()
        self.api_base_url.setReadOnly(True)

        self.api_http_language = QComboBox()
        self.api_http_language.addItems(["Python", "JavaScript", "curl"])
        self.api_http_language.currentTextChanged.connect(self.update_api_tab_content)

        self.api_http_snippet = QPlainTextEdit(readOnly=True)
        self.api_http_snippet.setFont(QFont("Consolas", 9))
        self.api_http_snippet.setLineWrapMode(QPlainTextEdit.NoWrap)

        self.api_tools = QPlainTextEdit(readOnly=True)
        self.api_tools.setFont(QFont("Consolas", 9))
        self.api_tools.setLineWrapMode(QPlainTextEdit.NoWrap)

        copy_base_btn = QPushButton("Copy")
        copy_base_btn.clicked.connect(lambda: self.copy_api_text(self.api_base_url.text(), "Base URL copied"))
        copy_http_btn = QPushButton("Copy")
        copy_http_btn.clicked.connect(
            lambda: self.copy_api_text(self.api_http_snippet.toPlainText(), "HTTP snippet copied")
        )
        copy_tools_btn = QPushButton("Copy")
        copy_tools_btn.clicked.connect(
            lambda: self.copy_api_text(self.api_tools.toPlainText(), "OpenAI tools JSON copied")
        )

        self.api_status = QLabel("")

        base_row = QHBoxLayout()
        base_row.addWidget(QLabel("Base URL"))
        base_row.addWidget(self.api_base_url, 1)
        base_row.addWidget(copy_base_btn)

        http_group = QGroupBox("HTTP in front of your LLM (saves tokens)")
        http_layout = QVBoxLayout(http_group)

        language_row = QHBoxLayout()
        language_row.addWidget(QLabel("Language"))
        language_row.addWidget(self.api_http_language, 1)
        http_layout.addLayout(language_row)

        snippet_row = QHBoxLayout()
        snippet_row.addWidget(QLabel("Request snippet"))
        snippet_row.addStretch(1)
        snippet_row.addWidget(copy_http_btn)
        http_layout.addLayout(snippet_row)
        http_layout.addWidget(self.api_http_snippet)

        tools_row = QHBoxLayout()
        tools_row.addWidget(QLabel("For agents that call HTTP tools directly"))
        tools_row.addStretch(1)
        tools_row.addWidget(copy_tools_btn)
        http_layout.addLayout(tools_row)
        http_layout.addWidget(self.api_tools)

        layout = QVBoxLayout(self.api_tab)
        layout.addLayout(base_row)
        layout.addWidget(http_group)
        layout.addWidget(self.api_status)

    def copy_api_text(self, text, message):
        QApplication.clipboard().setText(text)
        self.api_status.setText(message)

    def project_venv_python(self):
        root = Path(__file__).resolve().parents[1]
        return str((root / ".venv" / "Scripts" / "python.exe").resolve())

    def http_choice_payload(self, min_confidence):
        payload = {
            "state": "User says: package never arrived and wants a refund.",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Choose the support route.",
                    "criteria": {
                        "refund": "refund or chargeback request",
                        "shipping": "delivery delay or tracking issue",
                        "general": "everything else",
                    },
                }
            },
        }
        if min_confidence is not None:
            payload["min_confidence"] = round(float(min_confidence), 2)
        return payload

    def http_snippet_python(self, base_url, payload):
        payload_json = json.dumps(payload, indent=2)
        return "\n".join(
            [
                "import json",
                "",
                "try:",
                "    import httpx",
                "except ImportError:",
                "    httpx = None",
                "    import requests",
                "",
                f"BASE_URL = {json.dumps(base_url)}",
                "",
                "def call_online_llm(state: str) -> str:",
                "    return 'online_fallback'",
                "",
                f"payload = {payload_json}",
                "",
                "if httpx is not None:",
                "    response = httpx.post(f'{BASE_URL}/v1/systemone', json=payload, timeout=30)",
                "else:",
                "    response = requests.post(f'{BASE_URL}/v1/systemone', json=payload, timeout=30)",
                "response.raise_for_status()",
                "answer = response.json()['answers']['route']",
                "",
                "if answer.get('unsure') is True:",
                "    route = call_online_llm(payload['state'])",
                "else:",
                "    route = answer['choice']",
                "",
                "print('route:', route)",
                "print('confidence:', answer.get('confidence'))",
            ]
        )

    def http_snippet_javascript(self, base_url, payload):
        payload_json = json.dumps(payload, indent=2)
        return "\n".join(
            [
                f"const BASE_URL = {json.dumps(base_url)};",
                "",
                "function callOnlineLLM(state) {",
                "  return 'online_fallback';",
                "}",
                "",
                "async function main() {",
                f"  const payload = {payload_json};",
                "",
                "  const response = await fetch(`${BASE_URL}/v1/systemone`, {",
                "    method: 'POST',",
                "    headers: { 'Content-Type': 'application/json' },",
                "    body: JSON.stringify(payload),",
                "  });",
                "",
                "  if (!response.ok) {",
                "    throw new Error(`HTTP ${response.status}: ${await response.text()}`);",
                "  }",
                "",
                "  const answer = (await response.json()).answers.route;",
                "  const route = answer.unsure ? callOnlineLLM(payload.state) : answer.choice;",
                "",
                "  console.log('route:', route);",
                "  console.log('confidence:', answer.confidence);",
                "}",
                "",
                "main().catch((error) => {",
                "  console.error(error);",
                "  process.exitCode = 1;",
                "});",
            ]
        )

    def http_snippet_curl(self, base_url, payload):
        payload_json = json.dumps(payload, indent=2)
        return "\n".join(
            [
                "#!/usr/bin/env bash",
                f"BASE_URL={json.dumps(base_url)}",
                "",
                "call_online_llm() {",
                "  state=\"$1\"",
                "  echo \"online_fallback\"",
                "}",
                "",
                "payload=$(cat <<'JSON'",
                payload_json,
                "JSON",
                ")",
                "",
                "response=$(curl -sS -X POST \"$BASE_URL/v1/systemone\" -H \"Content-Type: application/json\" -d \"$payload\")",
                "",
                "route=$(RESPONSE=\"$response\" python - <<'PY'",
                "import json, os",
                "answer = json.loads(os.environ['RESPONSE'])['answers']['route']",
                "print('UNSURE' if answer.get('unsure') else answer.get('choice', ''))",
                "PY",
                ")",
                "",
                "if [ \"$route\" = \"UNSURE\" ]; then",
                "  route=\"$(call_online_llm \"User says: package never arrived and wants a refund.\")\"",
                "fi",
                "",
                "confidence=$(RESPONSE=\"$response\" python - <<'PY'",
                "import json, os",
                "print(json.loads(os.environ['RESPONSE'])['answers']['route'].get('confidence', ''))",
                "PY",
                ")",
                "",
                "printf 'route: %s\\n' \"$route\"",
                "printf 'confidence: %s\\n' \"$confidence\"",
            ]
        )

    def http_snippet_text(self, language, base_url, min_confidence):
        payload = self.http_choice_payload(min_confidence)
        if language == "JavaScript":
            return self.http_snippet_javascript(base_url, payload)
        if language == "curl":
            return self.http_snippet_curl(base_url, payload)
        return self.http_snippet_python(base_url, payload)

    def openai_tools_schema(self):
        min_conf = {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        }
        base_note = (
            "State-only context (max ~8K tokens, no memory). Returns probabilities and decided/defer; "
            "below min_confidence the answer value is \"unsure\" with unsure: true and raw_* fields."
        )
        return [
            {
                "type": "function",
                "function": {
                    "name": "jev_yesno",
                    "description": f"Binary check. {base_note}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "state": {"type": "string"},
                            "question": {"type": "string"},
                            "min_confidence": min_conf,
                        },
                        "required": ["state", "question"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "jev_choice",
                    "description": f"Pick one option. {base_note}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "state": {"type": "string"},
                            "question": {"type": "string"},
                            "options": {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                            },
                            "min_confidence": min_conf,
                        },
                        "required": ["state", "question", "options"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "jev_score",
                    "description": f"Ordinal score. {base_note}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "state": {"type": "string"},
                            "question": {"type": "string"},
                            "levels": {"type": "array", "items": {"type": "string"}},
                            "min_confidence": min_conf,
                        },
                        "required": ["state", "question", "levels"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "jev_batch",
                    "description": f"Many questions on one state. {base_note}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "state": {"type": "string"},
                            "questions": {"type": "object", "additionalProperties": True},
                            "min_confidence": min_conf,
                        },
                        "required": ["state", "questions"],
                    },
                },
            },
        ]

    def update_api_tab_content(self):
        if not hasattr(self, "api_base_url"):
            return

        base_url = self.base_url()
        self.api_base_url.setText(base_url)

        min_confidence = self.active_safety_threshold()
        language = self.api_http_language.currentText()
        self.api_http_snippet.setPlainText(self.http_snippet_text(language, base_url, min_confidence))

        self.api_tools.setPlainText(json.dumps(self.openai_tools_schema(), indent=2))

    def start_model_metadata_load(self):
        if self.metadata_worker and self.metadata_worker.isRunning():
            return

        entries = [(entry.name, entry.repo_id) for entry in self.model_entries if not is_jevk5(entry.name)]
        self.metadata_worker = ModelMetadataWorker(entries=entries, timeout_seconds=2.5)
        self.metadata_worker.done.connect(self.model_metadata_loaded)
        self.metadata_worker.finished.connect(self.model_metadata_finished)
        self.metadata_worker.start()

    def model_metadata_loaded(self, expected_files, expected_sizes):
        if expected_files:
            self.expected_files.update(expected_files)
        if expected_sizes:
            self.expected_sizes.update(expected_sizes)
        self.refresh_model_table()

    def model_metadata_finished(self):
        self.metadata_worker = None

    def model_changed(self, model_name):
        self.settings.setValue("model", model_name)
        self.update_gguf_quant_selector()
        self.refresh_model_table()
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

        if not self.host_is_valid_local(host):
            message = f"Invalid host: {host}"
            self.start_error_message = "Invalid host"
            self.log.appendPlainText(message)
            self.status.setText("Invalid host")
            return

        port_busy, bind_error = self.is_port_busy(host, port)
        if bind_error is not None and getattr(bind_error, "errno", None) != errno.EADDRINUSE:
            message = f"Invalid host: {host}"
            self.start_error_message = "Invalid host"
            self.log.appendPlainText(message)
            self.status.setText("Invalid host")
            return

        if port_busy:
            message = f"Port {port} is already in use on {host}."
            self.start_error_message = message
            self.log.appendPlainText(message)
            self.status.setText(message)
            return

        self.settings.setValue("host", host)
        self.settings.setValue("port", port)
        self.settings.setValue("model", model_name)

        device = self.selected_device()
        self.settings.setValue("device", device)
        self.last_start_device = device

        selected_quant = self.selected_gguf_quant(model_name)
        executable = str(executable_path()) if is_frozen_app() else sys.executable
        if device == "gpu":
            allowed, size = directml_allowed(model_name)
            if not allowed:
                limit_gb = DIRECTML_SIZE_LIMIT_BYTES / 1024**3
                model_gb = size / 1024**3
                message = (
                    f"{model_name} is {model_gb:.2f} GB, above the DirectML stability limit "
                    f"(~{limit_gb:.1f} GB on this machine). Use GPU (Vulkan) instead."
                )
                self.log.appendPlainText(message)
                self.status.setText(message)
                return

            gpu_python = self.gpu_runtime_python()
            if not gpu_python.exists():
                message = (
                    f"DirectML runtime not found at {gpu_python}. "
                    "Open Models tab and click Setup DirectML runtime."
                )
                self.log.appendPlainText(message)
                self.status.setText(message)
                answer = QMessageBox.question(
                    self,
                    "DirectML runtime missing",
                    "DirectML runtime is not installed yet. Open Models tab and run setup now?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if answer == QMessageBox.Yes:
                    self.tabs.setCurrentWidget(self.models_tab)
                    self.setup_gpu_runtime()
                return
            executable = str(gpu_python)
        elif device in {"vulkan", "cpu"}:
            try:
                server_path = find_llama_server(auto_download=False)
                self.log.appendPlainText(f"llama.cpp backend: {server_path}")
            except Exception:
                if is_frozen_app():
                    message = "Bundled llama.cpp runtime is missing. Reinstall Entschied."
                    self.log.appendPlainText(message)
                    self.status.setText(message)
                    QMessageBox.warning(self, "llama.cpp runtime", message)
                    return

                answer = QMessageBox.question(
                    self,
                    "llama.cpp runtime missing",
                    "llama.cpp backend is not installed. Download it now?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if answer != QMessageBox.Yes:
                    self.log.appendPlainText("Start cancelled: llama.cpp backend is missing.")
                    self.status.setText("llama.cpp runtime missing")
                    return

                try:
                    server_path = find_llama_server(log=lambda msg: self.log.appendPlainText(str(msg)), auto_download=True)
                    self.log.appendPlainText(f"llama.cpp backend ready: {server_path}")
                except Exception as exc:
                    message = f"Failed to download llama.cpp backend: {exc}"
                    self.log.appendPlainText(message)
                    self.status.setText(message)
                    QMessageBox.warning(self, "llama.cpp runtime", message)
                    return

            bundle = cached_gguf_bundle(model_name, selected_quant)
            rows = gguf_status(model_name)
            converted = next(
                (
                    row
                    for row in rows
                    if row.get("quant") == selected_quant and row.get("source") == "converted"
                ),
                None,
            )
            if not bool(bundle.get("ready")) and converted is None:
                if is_frozen_app():
                    message = f"No GGUF files found for {model_name} ({selected_quant}). Download from the GGUF repo first."
                else:
                    message = (
                        f"No GGUF files found for {model_name} ({selected_quant}). "
                        "Download from GGUF repo first, or convert locally."
                    )
                self.log.appendPlainText(message)
                self.status.setText(message)

                dialog = QMessageBox(self)
                dialog.setIcon(QMessageBox.Warning)
                dialog.setWindowTitle("GGUF missing")
                dialog.setText("llama.cpp needs GGUF files for the selected quant.")
                download_button = dialog.addButton("Download", QMessageBox.AcceptRole)
                convert_button = None
                if not is_frozen_app():
                    convert_button = dialog.addButton("Convert", QMessageBox.ActionRole)
                cancel_button = dialog.addButton("Cancel", QMessageBox.RejectRole)
                dialog.setDefaultButton(download_button)
                dialog.exec()

                clicked = dialog.clickedButton()
                if clicked == download_button:
                    self.tabs.setCurrentWidget(self.models_tab)
                    self.select_model_row(model_name)
                    self.download_selected()
                elif convert_button is not None and clicked == convert_button:
                    self.tabs.setCurrentWidget(self.models_tab)
                    self.select_model_row(model_name)
                    self.convert_selected_model()
                else:
                    _ = cancel_button
                return

        self.stop_requested = False
        self.server_ready = False
        self.has_seen_ready = False
        self.unexpected_stop_message = ""
        self.start_error_message = ""
        self.safety_hint_visible = False
        self.readiness_timer.setInterval(700)
        self.update_state()

        if is_frozen_app():
            args = [
                "--serve",
                "--model",
                model_name,
                "--device",
                device,
                "--host",
                host,
                "--port",
                str(port),
            ]
        else:
            args = [
                "-u",
                "-m",
                "entschied.serve",
                "--model",
                model_name,
                "--device",
                device,
                "--host",
                host,
                "--port",
                str(port),
            ]

        if device in {"vulkan", "cpu"}:
            args.extend(["--quant", selected_quant])

        threshold = self.active_safety_threshold()
        if threshold is not None:
            args.extend(["--min-confidence", f"{threshold:.2f}"])

        self.log.appendPlainText(f"> {executable} {' '.join(args)}")
        self.running_model_name = None
        self.proc.start(executable, args)
        if not self.proc.waitForStarted(4000):
            self.log.appendPlainText("Failed to start server process.")
            self.update_state()
            return

        self.running_model_name = model_name
        self.log.appendPlainText("Waiting for server readiness...")
        self.readiness_timer.start()

    def stop_server(self):
        if self.proc.state() == QProcess.NotRunning:
            return
        self.readiness_timer.stop()
        self.server_ready = False
        self.has_seen_ready = False
        self.unexpected_stop_message = ""
        self.start_error_message = ""
        self.stop_requested = True
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
                    creationflags=NO_WINDOW,
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
        if ready:
            if not self.server_ready:
                self.log.appendPlainText("Server is ready.")
            self.server_ready = True
            self.has_seen_ready = True
            if self.readiness_timer.interval() != 5000:
                self.readiness_timer.setInterval(5000)
        else:
            if self.server_ready:
                self.log.appendPlainText("Server readiness check failed.")
            self.server_ready = False
            if self.has_seen_ready and self.readiness_timer.interval() != 5000:
                self.readiness_timer.setInterval(5000)
        self.update_state()

    def update_state(self, *_):
        running = self.proc.state() != QProcess.NotRunning
        downloading = self.download_worker is not None and self.download_worker.isRunning()
        gpu_setup_running = self.gpu_setup_worker is not None and self.gpu_setup_worker.isRunning()
        convert_running = self.convert_worker is not None and self.convert_worker.isRunning()

        self.start_btn.setText("Stop" if running else "Start")
        self.host.setEnabled(not running)
        self.port.setEnabled(not running)
        self.model_combo.setEnabled(not running)
        self.device_combo.setEnabled(not running)

        request_running = self.worker is not None and self.worker.isRunning()
        self.send_btn.setEnabled(running and self.server_ready and not request_running)

        self.cancel_btn.setEnabled(downloading)
        self.download_btn.setEnabled(not downloading and not convert_running)
        self.delete_btn.setEnabled(not downloading and not convert_running)
        self.convert_btn.setEnabled(not convert_running and not downloading)
        self.setup_gpu_btn.setEnabled(not gpu_setup_running and not convert_running)
        if hasattr(self, "gguf_quant_combo"):
            show_quant = self.uses_gguf_downloads()
            self.gguf_quant_combo.setVisible(show_quant)
            self.gguf_quant_label.setVisible(show_quant)
            self.gguf_quant_combo.setEnabled(show_quant and self.gguf_quant_combo.count() > 1 and not downloading and not running)
            self.gguf_quant_label.setEnabled(show_quant and self.gguf_quant_combo.count() > 1 and not running)

        if hasattr(self, "tray_toggle_action"):
            self.tray_toggle_action.setText("Stop server" if running else "Start server")

        if running and self.server_ready:
            self.status.setText(f"Running on {self.base_url()}")
        elif running:
            if self.safety_hint_visible:
                self.status.setText("Safety threshold change will apply on next start")
            elif self.has_seen_ready:
                self.status.setText("Not responding")
            else:
                self.status.setText(f"Starting on {self.base_url()}")
        else:
            if self.unexpected_stop_message:
                self.status.setText(self.unexpected_stop_message)
                return
            if self.start_error_message:
                self.status.setText(self.start_error_message)
                return
            selected_model = self.selected_model_name()
            if self.model_is_downloaded(selected_model):
                self.status.setText("Stopped")
            else:
                self.status.setText(f"Stopped (download {selected_model} first)")

    def process_finished(self, exit_code, exit_status):
        self.readiness_timer.stop()
        was_ready = self.server_ready
        self.server_ready = False
        self.has_seen_ready = False
        self.running_model_name = None
        status_name = "normal" if exit_status == QProcess.NormalExit else "crashed"
        self.log.appendPlainText(f"Server process exited ({status_name}, code {exit_code}).")

        unexpected_exit = not self.stop_requested
        if unexpected_exit:
            self.unexpected_stop_message = f"Server stopped unexpectedly (exit code {exit_code})"
            self.log.appendPlainText(self.unexpected_stop_message)
        else:
            self.unexpected_stop_message = ""

        should_offer_cpu = (
            self.last_start_device in {"gpu", "vulkan"}
            and unexpected_exit
            and not was_ready
        )
        self.stop_requested = False
        self.safety_hint_visible = False
        self.update_state()

        if should_offer_cpu:
            answer = QMessageBox.question(
                self,
                "GPU start failed",
                "Selected GPU backend failed to start. Switch to CPU and retry now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer == QMessageBox.Yes:
                self.set_selected_device("cpu")
                QTimer.singleShot(0, self.start_server)

    def process_error(self, error):
        self.log.appendPlainText(f"Process error: {error}")

    def load_example(self, question_type):
        sample = EXAMPLES[question_type]
        self.state_in.setPlainText(sample["state"])
        self.instr_in.setText(sample["instructions"])
        self.criteria_in.setPlainText(sample["criteria"])
        self.criteria_in.setEnabled(question_type != "noul")

    def update_cutoff_note(self):
        if self.playground_cutoff_enabled.isChecked():
            text = (
                f"Playground requests use {self.playground_cutoff.value():.2f}. "
                "Other clients still get the Safety threshold."
            )
        elif self.safety_threshold_enabled.isChecked():
            text = (
                f"Playground requests use the Safety threshold "
                f"({self.safety_threshold_slider.value() / 100:.2f}) like any other client."
            )
        else:
            text = "Safety threshold is off, so every answer is returned as-is."
        self.cutoff_note.setText(text)

    def playground_cutoff_toggled(self, checked):
        self.settings.setValue("playground_cutoff_enabled", bool(checked))
        self.playground_cutoff.setEnabled(bool(checked))
        self.update_cutoff_note()

    def playground_cutoff_changed(self, value):
        self.settings.setValue("playground_min_confidence", float(value))
        self.update_cutoff_note()

    def update_safety_threshold_label(self, value):
        self.safety_threshold_value.setText(f"{float(value):.2f}")

    def active_safety_threshold(self):
        if not self.safety_threshold_enabled.isChecked():
            return None
        return float(self.safety_threshold_slider.value()) / 100.0

    def _note_safety_threshold_next_start(self):
        if self.proc.state() == QProcess.NotRunning:
            self.safety_hint_visible = False
            return
        self.status.setText("Safety threshold change will apply on next start")
        self.safety_hint_visible = True

    def safety_threshold_toggled(self, checked):
        self.settings.setValue("safety_threshold_enabled", bool(checked))
        self.safety_threshold_slider.setEnabled(bool(checked))
        self._note_safety_threshold_next_start()
        self.update_api_tab_content()
        self.update_cutoff_note()

    def safety_threshold_changed(self, slider_value):
        value = max(0.50, min(0.99, float(slider_value) / 100.0))
        self.update_safety_threshold_label(value)
        self.settings.setValue("safety_threshold", value)
        self._note_safety_threshold_next_start()
        self.update_api_tab_content()
        self.update_cutoff_note()


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
        if self.playground_cutoff_enabled.isChecked():
            payload["min_confidence"] = float(self.playground_cutoff.value())

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

    def _bind_sockaddr(self, family, sockaddr):
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(sockaddr)

    def host_is_valid_local(self, host):
        cleaned = str(host or "").strip().lower()
        if cleaned in {"0.0.0.0", "::", "[::]"}:
            return False

        try:
            addr_info = socket.getaddrinfo(host, 0, type=socket.SOCK_STREAM)
        except OSError:
            return False

        for family, _, _, _, sockaddr in addr_info:
            try:
                self._bind_sockaddr(family, sockaddr)
                return True
            except OSError:
                continue

        return False

    def is_port_busy(self, host, port):
        try:
            addr_info = socket.getaddrinfo(host, int(port), type=socket.SOCK_STREAM)
        except OSError as exc:
            return False, exc

        bind_succeeded = False
        last_error = None
        for family, _, _, _, sockaddr in addr_info:
            try:
                self._bind_sockaddr(family, sockaddr)
                bind_succeeded = True
            except OSError as exc:
                if getattr(exc, "errno", None) == errno.EADDRINUSE:
                    return True, exc
                last_error = exc

        if bind_succeeded:
            return False, None
        return False, last_error

    def model_is_downloaded(self, model_name):
        status = self.model_status_codes.get(model_name)
        if status is None:
            self.refresh_model_table()
            status = self.model_status_codes.get(model_name)
        return status in {"downloaded", "ready"}

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

        repo_id = str(self.download_state.get("repo_id", "")).strip()
        if not repo_id:
            model_name = self.download_state.get("model")
            entry = self.model_by_name.get(model_name)
            if entry is None:
                return
            repo_id = entry.repo_id

        patterns = [str(item) for item in self.download_state.get("allow_patterns", []) if item]
        if patterns:
            current_size = 0
            for filename in patterns:
                cached = try_to_load_from_cache(repo_id=repo_id, filename=filename, repo_type="model")
                if isinstance(cached, str):
                    path = Path(cached)
                    if path.exists():
                        current_size += path.stat().st_size
            current_size += self.incomplete_bytes(repo_id)
        else:
            _, folder = self.find_cached_repo(repo_id, repo_map)
            current_size = self.folder_size(folder)

        previous_size = int(self.download_state.get("downloaded", 0))
        now = time.time()
        previous_time = float(self.download_state.get("updated_at", now))

        if current_size > previous_size:
            self.download_state["downloaded"] = current_size

        self.download_state["updated_at"] = now
        self.show_download_info()

    def incomplete_bytes(self, repo_id):
        from huggingface_hub import constants as hf_constants

        folder = Path(hf_constants.HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--")) / "blobs"
        total = 0
        try:
            for item in folder.glob("*.incomplete"):
                total += item.stat().st_size
        except OSError:
            pass
        return total

    def show_download_info(self):
        state = self.download_state or {}
        if not state or self.download_worker is None:
            return
        model_name = state.get("model", "")
        downloaded = int(state.get("downloaded", 0))
        total = int(state.get("total", 0))
        speed = float(state.get("speed", 0.0) or 0.0)
        speed_part = f" • {human_size(speed)}/s" if speed > 0 else ""
        if total > 0:
            self.download_bar.setValue(max(0, min(100, int(downloaded * 100 / total))))
            text = f"Downloading {model_name}: {human_size(downloaded)} / {human_size(total)}{speed_part}"
        else:
            text = f"Downloading {model_name}: {human_size(downloaded)}{speed_part}"
        self.download_info.setText(text)

    def revision_note(self, repo_info):
        revisions = getattr(repo_info, "revisions", None) or []
        revision_count = len(revisions)
        if revision_count > 1:
            return f" • {revision_count} revisions cached"
        return ""

    def required_files_for_status(self, entry):
        required = {
            "tinyjev.json",
            "config.json",
            "head.safetensors",
            "tokenizer.json",
        }

        repo_key = normalize_repo_id(entry.repo_id)
        gguf_key = None
        try:
            gguf_key = normalize_repo_id(gguf_repo_id(entry.name))
        except Exception:
            gguf_key = None

        if gguf_key and repo_key == gguf_key:
            quant = self.selected_gguf_quant(entry.name)
            patterns = gguf_allow_patterns(entry.name, quant)
            gguf_name = next((name for name in patterns if str(name).lower().endswith(".gguf")), "")
            if gguf_name:
                required.add(str(gguf_name))
            return required

        required.add("model.safetensors")
        return required

    def model_status_for_vulkan(self, entry):
        quant = self.selected_gguf_quant(entry.name)
        bundle = cached_gguf_bundle(entry.name, quant)
        if bool(bundle.get("ready")):
            size = int(bundle.get("bytes", 0))
            return "downloaded", f"Downloaded ({human_size(size)})", size

        rows = gguf_status(entry.name)
        converted = next(
            (
                row
                for row in rows
                if row.get("quant") == quant and row.get("source") == "converted"
            ),
            None,
        )
        if converted is not None:
            size = int(converted.get("bytes", 0))
            return "ready", f"Converted ({human_size(size)})", size

        found = int(bundle.get("found", 0))
        if found > 0:
            size = int(bundle.get("bytes", 0))
            expected = int(bundle.get("expected", 0))
            return "partial", f"Partial ({found}/{expected} files, {human_size(size)})", size

        return "missing", "Not downloaded", 0

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

        if self.uses_gguf_downloads():
            return self.model_status_for_vulkan(entry)

        repo_info, folder = self.find_cached_repo(entry.repo_id, repo_map)

        if repo_info is not None:
            repo_size = int(repo_info.size_on_disk or 0)
            revision_note = self.revision_note(repo_info)

            expected_files = set(self.expected_files.get(entry.name, []))
            local_files = self.snapshot_file_names(repo_info)
            if expected_files:
                downloaded = expected_files.issubset(local_files)
            else:
                required_files = self.required_files_for_status(entry)
                downloaded = required_files.issubset(local_files)

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
            if is_jevk5(entry.name):
                size_value = gguf_expected_size(entry.name, self.selected_gguf_quant(entry.name))
            size_text = human_size(size_value) if size_value else "Unknown"

            status_code, status_text, disk_size = self.model_status_for_entry(entry, repo_map)
            self.model_status_codes[entry.name] = status_code
            self.model_status_text[entry.name] = status_text
            self.model_disk_sizes[entry.name] = disk_size

            gguf_rows = gguf_status(entry.name)
            if gguf_rows:
                gguf_parts = []
                for item in gguf_rows:
                    quant = str(item.get("quant"))
                    source = str(item.get("source", "not_downloaded"))
                    gguf_size_text = human_size(int(item.get("bytes", 0)))
                    if source == "downloaded":
                        gguf_parts.append(f"{quant}: downloaded ({gguf_size_text})")
                    elif source == "converted":
                        gguf_parts.append(f"{quant}: converted ({gguf_size_text})")
                    else:
                        gguf_parts.append(f"{quant}: not downloaded")
                gguf_text = ", ".join(gguf_parts)
            else:
                gguf_text = "-"

            self.models_table.setItem(row, 0, make_item(entry.name))
            self.models_table.setItem(row, 1, make_item(entry.params or "-"))
            self.models_table.setItem(row, 2, make_item(size_text))
            self.models_table.setItem(row, 3, make_item(entry.description or "-"))
            self.models_table.setItem(row, 4, make_item(status_text))
            self.models_table.setItem(row, 5, make_item(gguf_text))

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

        self.update_gguf_quant_selector()
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

        device = self.selected_device()
        repo_id = entry.repo_id
        allow_patterns = None
        quant = None

        if device in {"vulkan", "cpu"}:
            quant = self.selected_gguf_quant(model_name)
            repo_id = gguf_repo_id(model_name)
            allow_patterns = gguf_allow_patterns(model_name, quant)
            bundle = cached_gguf_bundle(model_name, quant)
            initial_bytes = int(bundle.get("bytes", 0))
            total_hint = 0
        else:
            initial_bytes = int(self.model_disk_sizes.get(model_name, 0))
            total_hint = int(self.expected_sizes.get(model_name, 0))

        self.download_worker = DownloadWorker(
            model_name=model_name,
            repo_id=repo_id,
            initial_bytes=initial_bytes,
            total_hint=total_hint,
            allow_patterns=allow_patterns,
        )
        self.download_worker.started_model.connect(self.download_started)
        self.download_worker.progress.connect(self.download_progress)
        self.download_worker.finished_model.connect(self.download_finished)
        self.download_worker.finished.connect(self.download_thread_done)

        self.download_result_received = False
        self.download_state = {
            "model": model_name,
            "repo_id": repo_id,
            "quant": quant,
            "allow_patterns": allow_patterns or [],
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
        prior = self.download_state or {}
        self.download_state = {
            "model": model_name,
            "repo_id": prior.get("repo_id", ""),
            "quant": prior.get("quant"),
            "allow_patterns": list(prior.get("allow_patterns", [])),
            "downloaded": downloaded,
            "total": total,
            "speed": 0.0,
            "updated_at": time.time(),
        }
        self.refresh_model_table()

    def download_progress(self, model_name, downloaded, total, speed):
        prior = self.download_state or {}
        downloaded = max(int(downloaded or 0), int(prior.get("downloaded", 0)) if prior.get("model") == model_name else 0)
        total = int(total or 0)
        if not speed and prior.get("model") == model_name:
            speed = float(prior.get("speed", 0.0) or 0.0)
        self.download_state = {
            "model": model_name,
            "repo_id": prior.get("repo_id", ""),
            "quant": prior.get("quant"),
            "allow_patterns": list(prior.get("allow_patterns", [])),
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
            self.download_info.setText("Cancelling...")

    def cache_repo_for_model(self, entry):
        if self.uses_gguf_downloads():
            return gguf_repo_id(entry.name)
        return entry.repo_id

    def delete_selected(self):
        model_name = self.current_selected_table_model()
        if not model_name:
            return
        entry = self.model_by_name.get(model_name)
        if entry is None:
            return

        if self.proc.state() != QProcess.NotRunning and model_name == self.running_model_name:
            QMessageBox.warning(
                self,
                "Model in use",
                f"Cannot delete {model_name} while the server is running with this model. Stop the server first.",
            )
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

        repo_id = self.cache_repo_for_model(entry)
        freed = self.delete_model_cache(repo_id)
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

        repo_id = self.cache_repo_for_model(entry)
        repo_info, folder = self.find_cached_repo(repo_id)
        target = Path(repo_info.repo_path) if repo_info is not None else folder

        if not target.exists():
            QMessageBox.information(self, "Folder missing", "No local cache folder exists for this model.")
            return

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def closeEvent(self, event):
        if self.is_quitting:
            self.cleanup_for_exit()
            event.accept()
            return

        if hasattr(self, "tray_icon") and self.tray_icon.isVisible():
            self.hide()
            event.ignore()
            return

        if not self.confirm_quit_if_busy():
            event.ignore()
            return

        self.is_quitting = True
        self.cleanup_for_exit()
        event.accept()


def main(argv=None):
    cli_args = list(sys.argv[1:] if argv is None else argv)
    start_minimized = "--minimized" in cli_args
    if start_minimized:
        cli_args = [arg for arg in cli_args if arg != "--minimized"]
    args = [sys.argv[0], *cli_args]

    username = getpass.getuser().strip() or "user"
    safe_username = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in username)
    instance_name = f"Entschied-{safe_username}"
    if SingleInstanceBridge.send_message(instance_name, "show"):
        return 0

    app = QApplication(args)
    app.setQuitOnLastWindowClosed(False)

    window = MainWindow()
    bridge = SingleInstanceBridge(instance_name, window.handle_single_instance_message, parent=window)
    if not bridge.listen():
        QMessageBox.warning(window, "Startup error", "Entschied is already running.")
        return 1
    window.single_instance_bridge = bridge

    if start_minimized:
        window.hide()
    else:
        window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
