from __future__ import annotations

import subprocess
import importlib.util
import sys
from pathlib import Path


def is_frozen_app() -> bool:
    return bool(getattr(sys, "frozen", False))


def executable_path() -> Path:
    return Path(sys.executable).resolve()


def app_root() -> Path:
    if is_frozen_app():
        return executable_path().parent
    return Path(__file__).resolve().parents[1]


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def torch_available() -> bool:
    return module_available("torch")


def directml_available() -> bool:
    return module_available("torch_directml")


# Console children of a windowed exe otherwise each get their own empty console window.
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
