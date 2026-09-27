# TinyJev Server GUI

A small desktop app for running a local [TinyJev](https://huggingface.co/AnkitAI/tinyjev-0.6b) server and trying requests against it.

- Start and stop the server, set host and port, pick the model to serve
- Model downloads manager (download/cancel/resume/delete/open cache folder) with status and size
- Live server log
- Playground for `noul` (yes/no), `choice` and `score` questions

## Install

```
python -m venv .venv
.venv\Scripts\activate
pip install -e .
tinyjev-gui
```

Or run without installing: `python -m tinyjev_gui`.

The first start downloads the model weights (about 1.2 GB) from Hugging Face into the usual HF cache.

## API

The server exposes `POST /v1/systemone`:

```json
{"state": "text to judge", "questions": {"result": {"type": "noul", "instructions": "Is this a refund request?"}}}
```

`choice` takes `criteria` as an object of option to description, `score` takes an ordered list of levels.

## License

MIT
