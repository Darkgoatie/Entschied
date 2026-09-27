from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableWidgetItem


@dataclass(frozen=True)
class ModelEntry:
    name: str
    repo_id: str
    params: str
    description: str


def normalize_repo_id(repo_id):
    return repo_id.strip().lower()


def human_size(num_bytes):
    size = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def make_item(text):
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item
