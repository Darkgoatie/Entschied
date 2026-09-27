import json
import socket
import subprocess
import sys

import httpx
from PySide6.QtCore import QProcess, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

MODEL = "TinyJev-0.6B"

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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TinyJev Server")
        self.resize(1000, 700)

        self.settings = QSettings("TinyJev", "TinyJev-Server-GUI")
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self.read_output)
        self.proc.stateChanged.connect(self.update_state)
        self.proc.finished.connect(self.process_finished)
        self.proc.errorOccurred.connect(self.process_error)

        self.readiness_timer = QTimer(self)
        self.readiness_timer.setInterval(700)
        self.readiness_timer.timeout.connect(self.poll_readiness)

        self.server_ready = False
        self.health_worker = None
        self.worker = None

        self.host = QLineEdit(self.settings.value("host", "127.0.0.1"))
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(int(self.settings.value("port", 8077)))

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.toggle_server)
        self.status = QLabel()

        top = QHBoxLayout()
        top.addWidget(QLabel("Host"))
        top.addWidget(self.host)
        top.addWidget(QLabel("Port"))
        top.addWidget(self.port)
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

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addLayout(top)
        layout.addWidget(split, 1)
        self.setCentralWidget(root)

        self.load_example("noul")
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

    def start_server(self):
        host = self.host.text().strip() or "127.0.0.1"
        port = self.port.value()
        if self.is_port_busy(host, port):
            message = f"Port {port} is already in use on {host}."
            self.log.appendPlainText(message)
            self.status.setText(message)
            return

        self.settings.setValue("host", host)
        self.settings.setValue("port", port)

        self.server_ready = False
        self.update_state()

        args = [
            "-u",
            "-m",
            "tinyjev.cli",
            "serve",
            MODEL,
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

        self.start_btn.setText("Stop" if running else "Start")
        self.host.setEnabled(not running)
        self.port.setEnabled(not running)

        request_running = self.worker is not None and self.worker.isRunning()
        self.send_btn.setEnabled(running and self.server_ready and not request_running)

        if running and self.server_ready:
            self.status.setText(f"Running on {self.base_url()}")
        elif running:
            self.status.setText(f"Starting on {self.base_url()}")
        else:
            self.status.setText("Stopped")

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

    def closeEvent(self, event):
        self.stop_server()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
