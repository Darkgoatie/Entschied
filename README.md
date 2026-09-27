# TinyJev Server GUI

Desktop app for running a local [TinyJev](https://huggingface.co/AnkitAI/tinyjev-0.6b) server and exposing it to other LLMs.

## Features

- Start/stop `tinyjev.cli serve` with host/port/model controls (default port: `8077`)
- Model manager: download/cancel/resume/delete/open cache folder
- Correct on-disk cache size display from Hugging Face cache repo totals (with revision count note)
- Playground for `noul` (yes/no), `choice`, and `score`
- API tab with copyable:
  - base URL
  - curl example
  - OpenAI function-calling tool schema JSON
  - MCP client config snippet (`mcpServers`)
- System tray mode: close window to tray, keep server running
- Startup options:
  - **Start server when app opens**
  - **Start app on login** (Windows Run key, starts minimized)

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

Or:

```bash
python -m tinyjev_gui
```

## API

The server endpoint is:

- `POST /v1/systemone`
- `GET /health`

OpenAPI spec is in `docs/openapi.yaml`.

Example request:

```json
{
  "state": "The user wrote: my order never arrived and I want my money back.",
  "questions": {
    "refund": {
      "type": "noul",
      "instructions": "Is this a refund request?"
    }
  }
}
```

`choice` uses `criteria` as an object (`option -> description`), `score` uses `criteria` as an ordered list of levels.

## MCP server

This repo includes a stdio MCP server entrypoint:

```bash
python -m tinyjev_gui.mcp
```

Console script (after install):

```bash
tinyjev-mcp
```

MCP tools:

- `jev_yesno(state, question)`
- `jev_choice(state, question, options)`
- `jev_score(state, question, levels)`
- `jev_batch(state, questions)`

By default it forwards to `http://127.0.0.1:8077`. Override with:

```bash
set TINYJEV_URL=http://127.0.0.1:8077
```

Use the API tab for ready-to-copy `mcpServers` JSON (Atomic Chat / Claude Desktop / Zed style config).

## Tray and login behavior

- Closing the main window hides it to the system tray (server keeps running).
- Tray menu: **Show**, **Start/Stop server**, **Quit**.
- **Quit** stops the TinyJev server before exit.
- On Windows, enabling **Start app on login** writes:
  - `HKCU\Software\Microsoft\Windows\CurrentVersion\Run\TinyJevServerGUI`
  - command: `.venv\Scripts\pythonw.exe -m tinyjev_gui --minimized`

## License

MIT
