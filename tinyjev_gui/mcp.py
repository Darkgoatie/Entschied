import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("tinyjev")


def _base_url() -> str:
    value = os.environ.get("TINYJEV_URL", "http://127.0.0.1:8077").strip()
    return value.rstrip("/") or "http://127.0.0.1:8077"


def _request_systemone(
    questions: dict[str, Any],
    state: str,
    min_confidence: float | None = None,
) -> dict[str, Any]:
    base_url = _base_url()
    url = f"{base_url}/v1/systemone"
    payload = {"state": state, "questions": questions}
    if min_confidence is not None:
        payload["min_confidence"] = min_confidence

    try:
        response = httpx.post(url, json=payload, timeout=45)
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"TinyJev server is not reachable at {base_url}. Start TinyJev Server GUI and click Start."
        ) from exc

    if response.status_code >= 400:
        try:
            details = response.json()
        except ValueError:
            details = response.text.strip() or "unknown error"
        raise RuntimeError(f"TinyJev request failed ({response.status_code}): {details}")

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("TinyJev returned non-JSON output.") from exc

    answers = data.get("answers")
    if not isinstance(answers, dict):
        raise RuntimeError("TinyJev response did not include an answers object.")
    return answers


@mcp.tool()
def jev_yesno(
    state: str,
    question: str,
    min_confidence: float | None = None,
) -> dict[str, Any]:
    """Binary route/gate check. State is the only context (max ~8K tokens, no memory). Returns probabilities and decided/defer; below threshold the answer is "unsure"."""

    answers = _request_systemone(
        questions={"result": {"type": "noul", "instructions": question}},
        state=state,
        min_confidence=min_confidence,
    )
    return answers["result"]


@mcp.tool()
def jev_choice(
    state: str,
    question: str,
    options: dict[str, str],
    min_confidence: float | None = None,
) -> dict[str, Any]:
    """Pick one option. State is the only context (max ~8K tokens, no memory). Returns option probabilities and decided/defer; below threshold the answer is "unsure"."""

    answers = _request_systemone(
        questions={
            "result": {
                "type": "choice",
                "instructions": question,
                "criteria": options,
            }
        },
        state=state,
        min_confidence=min_confidence,
    )
    return answers["result"]


@mcp.tool()
def jev_score(
    state: str,
    question: str,
    levels: list[str],
    min_confidence: float | None = None,
) -> dict[str, Any]:
    """Ordinal score. State is the only context (max ~8K tokens, no memory). Returns score probabilities and decided/defer; below threshold the answer is "unsure"."""

    answers = _request_systemone(
        questions={
            "result": {
                "type": "score",
                "instructions": question,
                "criteria": levels,
            }
        },
        state=state,
        min_confidence=min_confidence,
    )
    return answers["result"]


@mcp.tool()
def jev_batch(
    state: str,
    questions: dict[str, Any],
    min_confidence: float | None = None,
) -> dict[str, Any]:
    """Many questions, one state. State is the only context (max ~8K tokens, no memory). Returns probabilities and decided/defer; below threshold the answer is "unsure"."""

    return _request_systemone(
        questions=questions,
        state=state,
        min_confidence=min_confidence,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
