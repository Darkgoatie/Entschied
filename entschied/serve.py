from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import tinyjev

DEFAULT_MIN_CONFIDENCE = 0.70
MAX_BODY = 2_000_000


def _shutdown_agent_resources(agent) -> None:
    if agent is None:
        return

    backbone = getattr(agent, "backbone", None)
    shutdown = getattr(backbone, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception:
            pass

    handle = getattr(agent, "_llama_server_handle", None)
    if handle is not None:
        try:
            handle.stop()
        except Exception:
            pass


def _local_model_source(model_name: str) -> str:
    from .vulkan_setup import resolve_snapshot

    try:
        return str(resolve_snapshot(model_name, local_files_only=True))
    except Exception:
        raise FileNotFoundError(
            f"{model_name} is not downloaded for CPU/DirectML. "
            f"Download it in the Models tab with Device set to CPU or GPU (DirectML), or use --device vulkan."
        ) from None


def _llama_cpp_agent(model_name: str, mode: str, quant: str | None = None):
    import tinyjev.backends.torch_backend as torch_backend

    from .vulkan_backend import LlamaCppQwen3Backbone, LlamaServerHandle
    from .vulkan_setup import preferred_gguf, resolve_gguf_quant, resolve_gguf_snapshot

    selected_quant = resolve_gguf_quant(model_name, quant)
    load_source = model_name

    try:
        snapshot_root, gguf_path, selected_quant = resolve_gguf_snapshot(
            model_name,
            preferred_quant=selected_quant,
            local_files_only=True,
        )
        load_source = str(snapshot_root)
    except Exception:
        gguf_path = preferred_gguf(model_name, preferred_quant=selected_quant)

    host = os.environ.get("ENTSCHIED_LLAMA_HOST", "127.0.0.1")
    port = os.environ.get("ENTSCHIED_LLAMA_PORT", "").strip()
    handle = LlamaServerHandle.start(
        model_name=model_name,
        gguf_path=gguf_path,
        host=host,
        port=int(port) if port else None,
        mode="cpu" if mode == "cpu" else "vulkan",
    )

    os.environ["ENTSCHIED_LLAMA_URL"] = handle.base_url
    os.environ["ENTSCHIED_LLAMA_MODE"] = "cpu" if mode == "cpu" else "vulkan"
    os.environ["ENTSCHIED_MODEL_NAME"] = model_name
    os.environ["ENTSCHIED_GGUF_PATH"] = str(gguf_path)
    os.environ["ENTSCHIED_GGUF_QUANT"] = selected_quant

    torch_backend.Qwen3Backbone = LlamaCppQwen3Backbone

    try:
        agent = tinyjev.load(load_source, backend="torch", device="cpu")
    except Exception:
        handle.stop()
        raise

    agent._llama_server_handle = handle
    agent._llama_gguf_path = str(gguf_path)
    agent._llama_model_root = str(load_source)
    agent._llama_mode = mode
    return agent


def _jevk5_agent(model_name: str, mode: str, quant: str | None = None):
    from .jevk5 import JevK5Agent
    from .vulkan_backend import LlamaServerHandle
    from .vulkan_setup import gguf_source, resolve_gguf_snapshot

    _, gguf_path, selected_quant = resolve_gguf_snapshot(model_name, preferred_quant=quant, local_files_only=True)
    port = os.environ.get("ENTSCHIED_LLAMA_PORT", "").strip()
    handle = LlamaServerHandle.start(
        model_name=model_name,
        gguf_path=gguf_path,
        host=os.environ.get("ENTSCHIED_LLAMA_HOST", "127.0.0.1"),
        port=int(port) if port else None,
        mode="cpu" if mode == "cpu" else "vulkan",
        embeddings=False,
    )
    source = gguf_source(model_name)
    agent = JevK5Agent(
        name=model_name,
        base_url=handle.base_url,
        temperature=float(source["temperature"]),
        knockout_temperature=source.get("knockout_temperature"),
    )
    agent.backbone = agent
    agent._llama_server_handle = handle
    agent._llama_gguf_path = str(gguf_path)
    agent._llama_mode = mode
    print(f"{model_name} {selected_quant} on llama.cpp ({mode}), temperature {agent.temperature}", flush=True)
    return agent


def load_agent(model_name: str, device: str, quant: str | None = None):
    from .vulkan_setup import is_jevk5

    if is_jevk5(model_name):
        if device not in {"vulkan", "cpu"}:
            raise ValueError(f"{model_name} only runs on llama.cpp. Use --device vulkan or --device cpu.")
        return _jevk5_agent(model_name, mode=device, quant=quant)

    if device in {"vulkan", "cpu"}:
        return _llama_cpp_agent(model_name, mode=device, quant=quant)

    if device == "gpu":
        import torch_directml
        import tinyjev.backends.torch_backend as torch_backend

        from .gpu_backend import DirectMLQwen3Backbone
        from .vulkan_setup import directml_allowed

        allowed, size = directml_allowed(model_name)
        if not allowed:
            limit_gb = 5.5
            model_gb = size / 1024**3
            raise ValueError(
                f"DirectML is limited to about {limit_gb:.1f} GB model files on this machine; "
                f"{model_name} is {model_gb:.2f} GB. Use --device vulkan for this model."
            )

        source = _local_model_source(model_name)
        torch_backend.Qwen3Backbone = DirectMLQwen3Backbone
        dml_device = str(torch_directml.device())
        return tinyjev.load(source, backend="torch", device=dml_device)

    if device == "cpu_torch":
        return tinyjev.load(_local_model_source(model_name), backend="torch", device="cpu")

    raise ValueError(f"Unsupported device: {device}")


def _reject_nonfinite(value):
    raise ValueError("non-finite JSON literals are not accepted")


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _validate_min_confidence(value: Any, field_name: str) -> float:
    parsed = _as_float(value)
    if parsed is None:
        raise ValueError(f"{field_name} must be a finite number between 0 and 1")
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{field_name} must be in range [0, 1]")
    return parsed


def _confidence_for_answer(answer: dict[str, Any]) -> float | None:
    answer_type = answer.get("type")

    if answer_type == "choice":
        probabilities = answer.get("probabilities")
        if isinstance(probabilities, dict):
            values = [p for p in (_as_float(v) for v in probabilities.values()) if p is not None]
            if values:
                return max(values)
        return _as_float(answer.get("confidence"))

    if answer_type == "noul":
        probability = _as_float(answer.get("noul"))
        if probability is None:
            return None
        return max(probability, 1.0 - probability)

    if answer_type == "score":
        return _as_float(answer.get("confidence"))

    return None


def _resolve_confidence_cutoffs(
    body: dict[str, Any],
    server_min_confidence: float | None,
) -> tuple[float | None, dict[str, float]]:
    request_cutoff = None
    use_confidence_cutoff = body.get("use_confidence_cutoff")

    if use_confidence_cutoff is not None and not isinstance(use_confidence_cutoff, bool):
        raise ValueError("use_confidence_cutoff must be true or false")

    if "min_confidence" in body and body.get("min_confidence") is not None:
        request_cutoff = _validate_min_confidence(body.get("min_confidence"), "min_confidence")
    elif use_confidence_cutoff:
        request_cutoff = DEFAULT_MIN_CONFIDENCE
    elif server_min_confidence is not None:
        request_cutoff = server_min_confidence

    per_question_cutoffs: dict[str, float] = {}
    questions = body.get("questions")
    if isinstance(questions, dict):
        for question_id, question in questions.items():
            if not isinstance(question_id, str) or not isinstance(question, dict):
                continue
            if "min_confidence" in question and question.get("min_confidence") is not None:
                per_question_cutoffs[question_id] = _validate_min_confidence(
                    question.get("min_confidence"),
                    f"questions.{question_id}.min_confidence",
                )

    return request_cutoff, per_question_cutoffs


def _mark_unsure(answer: dict[str, Any]) -> None:
    answer_type = answer.get("type")

    if answer_type == "choice":
        answer["raw_choice"] = answer.get("choice")
        answer["choice"] = "unsure"
    elif answer_type == "noul":
        probability = _as_float(answer.get("noul"))
        if probability is not None:
            answer["raw_answer"] = "yes" if probability >= 0.5 else "no"
            answer["probabilities"] = {
                "yes": float(probability),
                "no": float(1.0 - probability),
            }
        answer["noul"] = "unsure"
    elif answer_type == "score":
        answer["raw_score"] = answer.get("score")
        answer["score"] = "unsure"

    answer["unsure"] = True


def _annotate_decisions(
    body: dict[str, Any],
    response: dict[str, Any],
    server_min_confidence: float | None,
) -> dict[str, Any]:
    request_cutoff, per_question_cutoffs = _resolve_confidence_cutoffs(body, server_min_confidence)
    if request_cutoff is None and not per_question_cutoffs:
        return response

    answers = response.get("answers")
    if not isinstance(answers, dict):
        return response

    for question_id, answer in answers.items():
        if not isinstance(question_id, str) or not isinstance(answer, dict):
            continue

        cutoff = per_question_cutoffs.get(question_id, request_cutoff)
        if cutoff is None:
            continue

        confidence = _confidence_for_answer(answer)
        decided = confidence is not None and confidence >= cutoff
        answer["decided"] = decided
        if decided:
            answer.pop("defer", None)
            answer.pop("unsure", None)
            answer.pop("raw_choice", None)
            answer.pop("raw_answer", None)
            answer.pop("raw_score", None)
        else:
            _mark_unsure(answer)
            answer["defer"] = True

    return response


def make_handler(agent, server_min_confidence: float | None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):
            pass

        def _send(self, code: int, data: dict):
            body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                raise ValueError(f"request body must be 1..{MAX_BODY} bytes")
            origin = self.headers.get("Origin")
            if origin:
                raise ValueError("cross-origin requests are disabled")
            return json.loads(self.rfile.read(length), parse_constant=_reject_nonfinite)

        def do_GET(self):
            if self.path == "/health":
                self._send(
                    200,
                    {
                        "ready": True,
                        "model": agent.name,
                        "family": agent.manifest["family"],
                        "backend": agent.backend,
                    },
                )
            elif self.path == "/v1/models":
                self._send(
                    200,
                    {
                        "data": [
                            {
                                "id": agent.name,
                                "family": agent.manifest["family"],
                                "backend": agent.backend,
                                "upstream": agent.manifest.get("upstream", {}),
                            }
                        ]
                    },
                )
            else:
                self._send(404, {"error": "unknown endpoint"})

        def do_POST(self):
            try:
                if self.path == "/predict":
                    self._send(200, agent.predict(self._body()))
                elif self.path == "/v1/systemone":
                    body = self._body()
                    result = agent.systemone(body)
                    self._send(200, _annotate_decisions(body, result, server_min_confidence))
                else:
                    self._send(404, {"error": "unknown endpoint"})
            except ValueError as exc:
                self._send(422, {"error": str(exc)})
            except Exception as exc:
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def serve(agent, host: str = "127.0.0.1", port: int = 8077, min_confidence: float | None = None):
    if min_confidence is not None:
        min_confidence = _validate_min_confidence(min_confidence, "--min-confidence")

    server = HTTPServer((host, port), make_handler(agent, min_confidence))
    cutoff_note = ""
    if min_confidence is not None:
        cutoff_note = f", default min_confidence={min_confidence:.2f}"
    print(
        f"entschied [{agent.name} on {agent.backend}] listening on http://{host}:{port} "
        f"(POST /predict, POST /v1/systemone{cutoff_note})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        _shutdown_agent_resources(agent)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Serve decision models with llama.cpp (CPU/Vulkan) or legacy torch modes")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=["cpu", "vulkan", "cpu_torch", "gpu"], default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--quant", default=None)
    parser.add_argument(
        "--min-confidence",
        nargs="?",
        const=DEFAULT_MIN_CONFIDENCE,
        type=float,
        default=None,
        help="Enable decided/defer output by default. If no value is given, uses 0.70.",
    )
    args = parser.parse_args(argv)

    try:
        min_confidence = None
        if args.min_confidence is not None:
            min_confidence = _validate_min_confidence(args.min_confidence, "--min-confidence")

        agent = load_agent(args.model, args.device, quant=args.quant)
    except Exception as exc:
        print(f"Failed to load {args.model} on {args.device}: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 2

    serve(agent, host=args.host, port=args.port, min_confidence=min_confidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
