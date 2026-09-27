import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

SERVER_GUIDANCE = (
    "State is the only context TinyJev sees. Keep state under about 8K tokens, "
    "include all facts needed for the decision, and do not rely on memory across calls. "
    "Outputs are calibrated probabilities: treat low confidence as uncertainty."
)

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
    """Yes/no decision from TinyJev.

    Use this for binary judgments. State is the only context TinyJev sees. Keep state
    under about 8K tokens, include everything needed, and do not assume memory between
    calls. TinyJev returns calibrated probabilities; low confidence should be treated
    as uncertainty.
    """

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
    """Categorical choice from TinyJev.

    Provide option->description in options. State is the only context TinyJev sees.
    Keep state under about 8K tokens, include everything needed, and do not assume
    memory between calls. TinyJev returns calibrated probabilities; low confidence
    means uncertain classification.
    """

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
    """Ordinal score from TinyJev.

    Provide ordered score levels from low to high. State is the only context TinyJev
    sees. Keep state under about 8K tokens, include everything needed, and do not
    assume memory between calls. TinyJev returns calibrated probabilities; low
    confidence means uncertain scoring.
    """

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
    """Run multiple TinyJev questions in one call on the same state.

    Pass questions exactly in TinyJev API shape: each entry has type/instructions and
    criteria for choice/score. State is the only context TinyJev sees. Keep state under
    about 8K tokens, include everything needed, and do not assume memory between calls.
    TinyJev outputs calibrated probabilities; treat low confidence as uncertainty.
    """

    return _request_systemone(
        questions=questions,
        state=state,
        min_confidence=min_confidence,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
