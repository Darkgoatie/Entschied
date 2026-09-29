"""JevK5 decision models served through llama-server's next-token log-probabilities.

Prompt, option mapping and the many-option readout follow the JevK5 runtime
(github.com/allebee/jevk5, Apache-2.0), which in turn follows SemIf (TheoLeeCJ/SemIf, MIT).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx

LETTERS = "ABCDEFGHIJKLMNOP"
MISSING_MARGIN = 2.0
KNOCKOUT_DEFAULT = 0.77
SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

Reader = Callable[[list[str]], Sequence[float]]


def prompt_text(state, criterion: str, options: list[str]) -> str:
    payload = {
        "evidence": state,
        "criterion": criterion,
        "options": [{"letter": LETTERS[i], "description": d} for i, d in enumerate(options)],
    }
    return CHAT_TEMPLATE.format(system=SYSTEM, user=json.dumps(payload, ensure_ascii=False))


def normalized(question: dict) -> dict:
    if not isinstance(question, dict):
        raise ValueError("each question must be an object")
    kind = question.get("type")
    if kind == "boolean":
        kind = "noul"
    if kind not in ("noul", "choice", "score"):
        raise ValueError(f"unknown question type {question.get('type')!r}")
    if not question.get("instructions"):
        raise ValueError("question is missing instructions")
    criteria = question.get("criteria")
    if kind == "choice":
        if isinstance(criteria, list):
            criteria = dict.fromkeys(str(c) for c in criteria)
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("choice criteria must name at least two options")
    if kind == "score" and (not isinstance(criteria, list) or len(criteria) < 2):
        raise ValueError("score criteria must list at least two levels")
    return {**question, "type": kind, "criteria": criteria}


def decision_options(question: dict) -> list[tuple[str, str]]:
    crit = question.get("criteria")
    if question["type"] == "noul":
        pairs = [(k, (crit or {}).get(k) or f"The proposition is {k}.") for k in ("true", "false")]
    elif question["type"] == "choice":
        pairs = [(k, v or k) for k, v in crit.items()]
    else:
        pairs = [(str(i), level) for i, level in enumerate(crit)]
    return [(k, f"{k}: {d}") for k, d in pairs]


def _groups(n: int, count: int) -> list[range]:
    base, extra = divmod(n, count)
    runs, start = [], 0
    for g in range(count):
        stop = start + base + (g < extra)
        runs.append(range(start, stop))
        start = stop
    return runs


def _combine(read: Reader, texts: list[str]) -> list[float]:
    if len(texts) <= len(LETTERS):
        return list(read(texts))
    weights = _knockout(read, texts)
    total = sum(weights)
    return [w / total for w in weights]


def _knockout(read: Reader, texts: list[str]) -> list[float]:
    runs = _groups(len(texts), -(-len(texts) // len(LETTERS)))
    inner = [list(read([texts[i] for i in run])) for run in runs]
    inner = [[q / sum(p) for q in p] for p in inner]
    keep = max(1, len(LETTERS) // len(runs))
    ranked = [sorted(range(len(p)), key=lambda j: -p[j]) for p in inner]
    chosen = {(g, j) for g, order in enumerate(ranked) for j in order[:keep]}
    rest = sorted(
        ((g, j) for g, order in enumerate(ranked) for j in order[keep:]),
        key=lambda gj: -inner[gj[0]][gj[1]],
    )
    chosen.update(rest[: max(0, len(LETTERS) - len(chosen))])
    tops = [sorted(j for h, j in chosen if h == g) for g in range(len(runs))]
    final = _combine(read, [texts[run[j]] for run, top in zip(runs, tops) for j in top])
    shares, at = [], 0
    for top in tops:
        shares.append(dict(zip(top, final[at : at + len(top)])))
        at += len(top)
    in_final = sum(sum(f.values()) * sum(p[j] for j in f) for p, f in zip(inner, shares))
    weights = []
    for p, f in zip(inner, shares):
        mass = sum(f.values())
        weights += [f[j] * in_final if j in f else mass * q for j, q in enumerate(p)]
    return weights


def spread(read: Reader, texts: list[str], temperature: float) -> list[float]:
    if len(texts) <= len(LETTERS):
        return list(read(texts))
    probs = _combine(read, texts)
    if temperature != 1.0:
        probs = [q ** (1 / temperature) for q in probs]
        total = sum(probs)
        probs = [q / total for q in probs]
    return probs


class JevK5Agent:
    """Answers /v1/systemone and /predict requests with a JevK5 GGUF on llama-server."""

    backend = "llama.cpp"

    def __init__(self, name: str, base_url: str, temperature: float, knockout_temperature: float | None):
        self.name = name
        self.url = base_url.rstrip("/")
        self.temperature = float(temperature)
        self.knockout_temperature = float(knockout_temperature or KNOCKOUT_DEFAULT)
        self.manifest = {"family": "jevk5", "upstream": {"repo": "alibiserikbay/JevK5-GGUF"}}
        self.client = httpx.Client(timeout=600)

    def _post(self, path: str, payload: dict) -> dict:
        response = self.client.post(self.url + path, json=payload)
        response.raise_for_status()
        return response.json()

    def _letter_logprobs(self, prompt: str) -> tuple[dict[str, float], int]:
        tokens = self._post("/tokenize", {"content": prompt, "add_special": False, "parse_special": True})["tokens"]
        out = self._post(
            "/completion",
            {"prompt": tokens, "n_predict": 1, "n_probs": 40, "temperature": 0, "cache_prompt": False},
        )
        top = out["completion_probabilities"][0]["top_logprobs"]
        return {entry["token"]: entry["logprob"] for entry in top}, int(out.get("tokens_evaluated", 0))

    def probabilities(self, state, question: dict) -> tuple[dict[str, float], int]:
        options = decision_options(question)
        used = 0

        def read(texts: list[str]) -> list[float]:
            nonlocal used
            seen, count = self._letter_logprobs(prompt_text(state, question["instructions"], texts))
            used += count
            floor = min(seen.values(), default=0.0) - MISSING_MARGIN
            logprobs = [seen.get(LETTERS[i], floor) for i in range(len(texts))]
            top = max(logprobs)
            weights = [math.exp((z - top) / self.temperature) for z in logprobs]
            total = sum(weights)
            return [w / total for w in weights]

        probs = spread(read, [text for _, text in options], self.knockout_temperature)
        return {key: v for (key, _), v in zip(options, probs)}, used

    def decide(self, state, question: dict) -> dict[str, Any]:
        question = normalized(question)
        probs, tokens = self.probabilities(state, question)
        kind = question["type"]
        out: dict[str, Any] = {"type": kind, "confidence": max(probs.values()), "input_tokens": tokens}
        if kind == "noul":
            out["noul"] = round(probs["true"], 4)
        elif kind == "choice":
            out.update(choice=max(probs, key=probs.get), probabilities=probs)
        else:
            legend = {str(i): level for i, level in enumerate(question["criteria"])}
            out.update(score=round(sum(int(k) * v for k, v in probs.items()), 4), probabilities=probs, legend=legend)
        return out

    def _questions(self, body: dict) -> tuple[Any, dict]:
        if not isinstance(body, dict) or "state" not in body:
            raise ValueError("request needs a state")
        questions = body.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("request needs a questions object")
        return body["state"], questions

    def systemone(self, body: dict) -> dict:
        state, questions = self._questions(body)
        started = time.perf_counter()
        answers = {qid: self.decide(state, q) for qid, q in questions.items()}
        tokens = sum(a.pop("input_tokens") for a in answers.values())
        return {
            "model": body.get("model") or self.name,
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
            "latency_ms": round((time.perf_counter() - started) * 1e3, 2),
        }

    def predict(self, body: dict) -> dict:
        result = self.systemone(body)
        answers = {}
        for qid, a in result["answers"].items():
            item = dict(a)
            if a["type"] == "noul":
                item["p_true"] = a["noul"]
            answers[qid] = item
        return {"states": [{"answers": answers}], "execution": {"model_ms": result["latency_ms"]}}

    def shutdown(self) -> None:
        self.client.close()
