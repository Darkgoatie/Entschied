# TinyJev Server GUI

Desktop app for running a local [TinyJev](https://huggingface.co/AnkitAI/tinyjev-0.6b) server and using it as a fast decision layer.

## TinyJev in front of your LLM (recommended)

The main cost/latency win is to call TinyJev **before** your online LLM.
Use it to route, gate, skip, or choose a model/tool. If TinyJev is not confident, defer.

```python
import httpx

TINYJEV_URL = "http://127.0.0.1:8077"

state = "User says: package never arrived and wants a refund"

payload = {
    "state": state,
    "min_confidence": 0.85,
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "Where should this go?",
            "criteria": {
                "refund": "refund or chargeback request",
                "shipping": "delivery delay or tracking issue",
                "general": "everything else",
            },
        }
    },
}

answer = httpx.post(f"{TINYJEV_URL}/v1/systemone", json=payload, timeout=30).json()["answers"]["route"]

if answer["decided"]:
    # local decision path (0 online tokens for this decision)
    route = answer["choice"]
    confidence = answer["confidence"]
else:
    # defer to online LLM only when TinyJev is not confident
    # call_your_online_llm(state)
    pass
```

Why this saves tokens:
- decided locally: no online model call for that decision
- deferred: you pay online tokens only on uncertain cases

When MCP can add tokens instead of saving:
- tool schemas are sent each turn
- the model emits tool-call arguments (extra output tokens)
- the tool result comes back as extra input tokens
- the model needs another turn for the final answer

MCP is still useful for integration convenience, but token savings are usually best when TinyJev is called directly by app logic first.

## Features

- Start/stop local server with host/port/model controls (default `8077`)
- Device switch: CPU or DirectML GPU runtime
- Model manager: download/cancel/delete/open cache folder
- Playground for `noul`, `choice`, and `score`
- Optional confidence cutoff in playground (`decided` / `defer`)
- API tab with copyable base URL, curl sample, tool schema JSON, MCP config snippet
- System tray mode and startup options on Windows

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Run GUI:

```bash
tinyjev-gui
```

or:

```bash
python -m tinyjev_gui
```

## GPU runtime (DirectML)

Setup options:

- In **Models** tab, click **Setup GPU runtime**
- Or run:

```bash
python -m tinyjev_gui.gpu_setup
```

Runtime path:

- `%LOCALAPPDATA%/TinyJev/gpu-runtime`

Direct server entrypoint:

```bash
python -m tinyjev_gui.serve --model TinyJev-0.6B --device gpu --host 127.0.0.1 --port 8091
```

## API

Endpoints:
- `POST /v1/systemone`
- `GET /health`

OpenAPI spec: `docs/openapi.yaml`

Request fields for confidence gating:
- top-level `min_confidence` (0..1)
- top-level `use_confidence_cutoff: true` (uses default 0.85)
- per-question `min_confidence` override

Behavior:
- if no cutoff is provided and server has no `--min-confidence`, responses are unchanged
- with cutoff active, each answer includes:
  - `decided: true|false`
  - `defer: true` when `decided` is false

Server-wide default cutoff:

```bash
python -m tinyjev_gui.serve --model TinyJev-0.6B --device cpu --min-confidence
```

(or pass a value, e.g. `--min-confidence 0.9`)

## MCP server (secondary pattern)

Run:

```bash
python -m tinyjev_gui.mcp
```

Console script:

```bash
tinyjev-mcp
```

Tools:
- `jev_yesno(state, question, min_confidence=None)`
- `jev_choice(state, question, options, min_confidence=None)`
- `jev_score(state, question, levels, min_confidence=None)`
- `jev_batch(state, questions, min_confidence=None)`

Tool semantics (all four):
- `state` is the only context (keep under ~8K tokens, no cross-call memory)
- answers are probabilities
- branch on `decided` / `defer`

Default TinyJev URL is `http://127.0.0.1:8077`.
Override with `TINYJEV_URL`.

## Tray and login behavior

- Closing the window hides it to tray; server keeps running
- Tray menu: **Show**, **Start/Stop server**, **Quit**
- **Quit** stops the TinyJev server before exit
- Windows **Start app on login** writes:
  - `HKCU\Software\Microsoft\Windows\CurrentVersion\Run\TinyJevServerGUI`
  - command: `.venv\Scripts\pythonw.exe -m tinyjev_gui --minimized`

## License

MIT
