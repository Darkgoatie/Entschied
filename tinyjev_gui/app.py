import importlib
import sys

from .window import MainWindow, main as window_main

__all__ = ["MainWindow", "main"]


def run_self_test() -> int:
    modules = [
        "tinyjev_gui.common",
        "tinyjev_gui.runtime",
        "tinyjev_gui.gpu_setup",
        "tinyjev_gui.vulkan_setup",
        "tinyjev_gui.vulkan_backend",
        "tinyjev_gui.workers",
        "tinyjev_gui.serve",
        "tinyjev_gui.window",
    ]
    failures: list[tuple[str, Exception]] = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append((name, exc))

    if failures:
        for name, exc in failures:
            print(f"self-test import failed: {name}: {exc}", file=sys.stderr)
        return 1

    if getattr(sys, "frozen", False):
        import subprocess

        from .vulkan_setup import bundled_llama_server

        server = bundled_llama_server()
        if server is None:
            print("self-test failed: bundled llama-server.exe not found", file=sys.stderr)
            return 1
        result = subprocess.run([str(server), "--version"], capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"self-test failed: {server} --version exited {result.returncode}", file=sys.stderr)
            return 1
        print(f"llama-server: {server}")

    print("self-test ok")
    return 0


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--self-test":
        return run_self_test()
    if args and args[0] == "--serve":
        from .serve import main as serve_main

        return serve_main(args[1:])
    return window_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
