from __future__ import annotations

import argparse
import traceback

import tinyjev


def load_agent(model_name: str, device: str):
    if device == "gpu":
        import torch_directml
        import tinyjev.backends.torch_backend as torch_backend

        from .gpu_backend import DirectMLQwen3Backbone

        torch_backend.Qwen3Backbone = DirectMLQwen3Backbone
        dml_device = str(torch_directml.device())
        return tinyjev.load(model_name, backend="torch", device=dml_device)

    return tinyjev.load(model_name, backend="torch", device="cpu")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Serve TinyJev with CPU or DirectML GPU backend")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8077)
    args = parser.parse_args(argv)

    try:
        agent = load_agent(args.model, args.device)
    except Exception as exc:
        print(f"Failed to load {args.model} on {args.device}: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 2

    from tinyjev.serve import serve

    serve(agent, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
