import json
import sys

import httpx
from PySide6.QtCore import QProcess, QSettings, Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QPlainTextEdit, QPushButton, QSpinBox, QSplitter,
    QVBoxLayout, QWidget,
)

MODEL = "tinyjev-0.6b"

EXAMPLES = {
    "noul": {"state": "The user wrote: my order never arrived and I want my money back.",
             "instructions": "Is the user asking for a refund?", "criteria": ""},
    "choice": {"state": "The app crashes when I open settings.",
               "instructions": "Which team should handle this ticket?",
               "criteria": json.dumps({"bug": "software defect", "billing": "payment issue",
                                       "question": "how-to question"}, indent=2)},
    "score": {"state": "Thanks, that fixed it immediately!",
              "instructions": "How satisfied is the user?",
              "criteria": json.dumps(["very unhappy", "unhappy", "neutral", "happy", "very happy"], indent=2)},
}


class RequestWorker(QThread):
    done = Signal(str)

    def __init__(self, url, payload):
        super().__init__()
        self.url, self.payload = url, payload

    def run(self):
        try:
            r = httpx.post(self.url, json=self.payload, timeout=120)
            try:
                self.done.emit(json.dumps(r.json(), indent=2))
            except ValueError:
                self.done.emit(f"HTTP {r.status_code}\n{r.text}")
        except Exception as e:
            self.done.emit(f"Request failed: {e}")


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
        lay = QVBoxLayout(root)
        lay.addLayout(top)
        lay.addWidget(split, 1)
        self.setCentralWidget(root)

        self.load_example("noul")
        self.update_state()

    def base_url(self):
        host = self.host.text().strip() or "127.0.0.1"
        if host == "0.0.0.0":
            host = "127.0.0.1"
        return f"http://{host}:{self.port.value()}"

    def toggle_server(self):
        if self.proc.state() != QProcess.NotRunning:
            self.proc.terminate()
            if not self.proc.waitForFinished(5000):
                self.proc.kill()
            return
        self.settings.setValue("host", self.host.text())
        self.settings.setValue("port", self.port.value())
        args = ["-u", "-m", "tinyjev.cli", "serve", MODEL,
                "--host", self.host.text().strip(), "--port", str(self.port.value())]
        self.log.appendPlainText(f"> {sys.executable} {' '.join(args)}")
        self.proc.start(sys.executable, args)

    def read_output(self):
        text = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        self.log.appendPlainText(text.rstrip())

    def update_state(self, *_):
        running = self.proc.state() != QProcess.NotRunning
        self.start_btn.setText("Stop" if running else "Start")
        self.host.setEnabled(not running)
        self.port.setEnabled(not running)
        self.status.setText(f"Running on {self.base_url()}" if running else "Stopped")

    def load_example(self, t):
        ex = EXAMPLES[t]
        self.state_in.setPlainText(ex["state"])
        self.instr_in.setText(ex["instructions"])
        self.criteria_in.setPlainText(ex["criteria"])
        self.criteria_in.setEnabled(t != "noul")

    def send(self):
        q = {"type": self.qtype.currentText(), "instructions": self.instr_in.text()}
        if q["type"] != "noul":
            try:
                q["criteria"] = json.loads(self.criteria_in.toPlainText())
            except ValueError as e:
                self.result.setPlainText(f"Criteria is not valid JSON: {e}")
                return
        payload = {"state": self.state_in.toPlainText(), "questions": {"result": q}}
        self.send_btn.setEnabled(False)
        self.result.setPlainText("Waiting...")
        self.worker = RequestWorker(self.base_url() + "/v1/systemone", payload)
        self.worker.done.connect(self.show_result)
        self.worker.start()

    def show_result(self, text):
        self.result.setPlainText(text)
        self.send_btn.setEnabled(True)

    def closeEvent(self, e):
        if self.proc.state() != QProcess.NotRunning:
            self.proc.terminate()
            if not self.proc.waitForFinished(5000):
                self.proc.kill()
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
