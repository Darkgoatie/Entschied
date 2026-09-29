# Entschied

Run decision-maker AI models locally.

Supported models:

- TinyJev-0.6B, TinyJev-4B (pointer head on Qwen3)
- JevK5-2B, JevK5-4B, JevK5-9B from [alibiserikbay/JevK5-GGUF](https://huggingface.co/alibiserikbay/JevK5-GGUF) (Apache-2.0). These read answer-letter log-probabilities from llama-server and apply each file's calibration temperature. They run on llama.cpp only (CPU or Vulkan).

Desktop app for running a local [TinyJev](https://huggingface.co/AnkitAI/tinyjev-0.6b) server and using it as a fast decision layer.

## Download

- Installer: `Entschied-Setup-0.1.0.exe` (Windows 10/11 x64)
- The installer is unsigned. If SmartScreen blocks it, click **More info** then **Run anyway**.
- First run downloads the TinyJev model files from Hugging Face.
- The frozen app uses `llama.cpp` for both **GPU (Vulkan)** and **CPU**.
- 42-case parity vs PyTorch CPU baseline:
  - llama.cpp CPU: 42/42 same answers, max gap 0.020, mean gap 0.006
  - llama.cpp Vulkan: 42/42 same answers, max gap 0.018, mean gap 0.007

## TinyJev in front of your LLM (recommended)

The main cost/latency win is to call TinyJev **before** your online LLM.
Use it to route, gate, skip, or choose a model/tool. If TinyJev is not confident, defer.

```python
import httpx

ENTSCHIED_URL = "http://127.0.0.1:8077"

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

answer = httpx.post(f"{ENTSCHIED_URL}/v1/systemone", json=payload, timeout=30).json()["answers"]["route"]

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

## Connecting an agent

Use the **API** tab in the GUI.

- **HTTP in front of your LLM (recommended):** pick Python, JavaScript, or curl and copy the ready-to-run snippet.
- The snippet calls `POST /v1/systemone` with one choice question and falls back to your online LLM when TinyJev returns `unsure: true`.
- When the safety threshold is enabled, the snippet includes `min_confidence` automatically.
- For agents that call HTTP tools directly, copy the OpenAI tool schema JSON from the same tab.

## Features

- Start/stop local server with host/port/model controls (default `8077`)
- Device switch: CPU (llama.cpp) and GPU (Vulkan via llama.cpp)
- Source-only legacy modes (when torch is available): CPU (PyTorch) and GPU (DirectML)
- Model manager: download/cancel/delete/open cache folder, GGUF quant selector, optional GGUF conversion
- Playground for `noul`, `choice`, and `score`
- Safety threshold slider (0.50-0.99) with persisted value and server start integration
- Optional confidence cutoff in playground (`decided` / `defer`)
- API tab with copyable base URL, HTTP snippets (Python/JavaScript/curl), and OpenAI tool schema JSON
- System tray mode and startup options on Windows

## From source

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Building the installer

Requires Inno Setup 6 (`ISCC.exe`) and the project venv at `.venv`.

```bash
python build.py
```

`build.py` builds the frozen app, runs `Entschied.exe --self-test`, and creates `dist/Entschied-Setup-0.1.0.exe`.

Run GUI:

```bash
entschied
```

or:

```bash
python -m entschied
```

## GPU runtime (DirectML)

Setup options:

- In **Models** tab, click **Setup DirectML runtime**
- Or run:

```bash
python -m entschied.gpu_setup
```

Runtime path:

- `%LOCALAPPDATA%/Entschied/gpu-runtime`

DirectML is stable only for smaller checkpoints on this iGPU path. Models above ~5.5 GB (for example TinyJev-4B) are blocked in the GUI with a clear message; use Vulkan for those.

Direct server entrypoint:

```bash
python -m entschied.serve --model TinyJev-0.6B --device gpu --host 127.0.0.1 --port 8091
```

## GPU runtime (Vulkan / llama.cpp)

Entschied Vulkan mode runs the Qwen3 backbone through `llama-server` embeddings (`--embeddings --pooling none`) and keeps the TinyJev pointer head in Python.

- Frozen app bundles `llama-server.exe` and required DLLs under `llama/`
- Override executable: set `ENTSCHIED_LLAMA_SERVER`
- From source (non-frozen), if no local executable is found the app can download a Windows Vulkan release into `%LOCALAPPDATA%/Entschied/llama.cpp`
- GGUF files are downloaded from:
  - https://huggingface.co/darkgoatie/TinyJev-0.6B-GGUF
  - https://huggingface.co/darkgoatie/TinyJev-4B-GGUF
- Downloads pull only the selected quant plus support files (`head.safetensors`, `tinyjev.json`, `config.json`, tokenizer files, `README.md`)
- GGUF files can also live in `ENTSCHIED_GGUF_DIR` when set
- Local conversion is optional fallback only (source mode)

In **Models** tab, pick a GGUF quant and click **Download**. In the frozen app, downloads from the GGUF repos are the model path.

Direct server entrypoint:

```bash
python -m entschied.serve --model TinyJev-4B --device vulkan --host 127.0.0.1 --port 8096
```

CPU llama.cpp entrypoint:

```bash
python -m entschied.serve --model TinyJev-0.6B --device cpu --host 127.0.0.1 --port 8097
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
  - below threshold, answer value becomes `"unsure"` with `unsure: true` and raw value in `raw_choice` / `raw_answer` / `raw_score`

Server-wide default cutoff:

```bash
python -m entschied.serve --model TinyJev-0.6B --device cpu --min-confidence
```

(or pass a value, e.g. `--min-confidence 0.9`)

## MCP server (secondary pattern)

Run:

```bash
python -m entschied.mcp
```

Console script:

```bash
entschied-mcp
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

Default Entschied URL is `http://127.0.0.1:8077`.
Override with `ENTSCHIED_URL`.

## Tray and login behavior

- Closing the window hides it to tray; server keeps running
- Tray menu: **Show**, **Start/Stop server**, **Quit**
- **Quit** stops the Entschied server before exit
- Windows **Start app on login** writes:
  - `HKCU\Software\Microsoft\Windows\CurrentVersion\Run\Entschied`
  - command while running from source: `.venv\Scripts\pythonw.exe -m entschied --minimized`
  - command in the frozen app: `Entschied.exe --minimized`

## License

MIT
