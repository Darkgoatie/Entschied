import argparse
import asyncio
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_token_counter():
    try:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        return lambda s: len(enc.encode(s)), "tiktoken:o200k_base"
    except Exception:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
        return lambda s: len(tok.encode(s)), "qwen-tokenizer-approx"


def load_script_namespace(path: Path, stop_marker: str):
    text = path.read_text(encoding="utf-8")
    if stop_marker not in text:
        raise ValueError(f"{path} missing marker: {stop_marker}")
    prefix = text.split(stop_marker, 1)[0]
    ns = {}
    exec(prefix, ns)
    return ns


def load_cases(approach_script: Path, dev_script: Path, include_dev: bool):
    approach_ns = load_script_namespace(approach_script, "rows = []")
    cases = []

    for idx, item in enumerate(approach_ns["cases"], start=1):
        state, qtype, question, expected, criteria = item
        cases.append(
            {
                "id": f"approach-{idx:03d}",
                "dataset": "approach",
                "type": qtype,
                "state": state,
                "question": question,
                "expected": str(expected),
                "criteria": criteria,
            }
        )

    if include_dev and dev_script.exists():
        dev_ns = load_script_namespace(dev_script, "rows = []")
        chooser = dev_ns["C"]
        for idx, item in enumerate(dev_ns["cases"], start=1):
            state, qtype, question, expected = item[:4]
            criteria = None
            if qtype == "choice":
                criteria = chooser[item[4]]
            elif qtype == "score":
                criteria = item[4]
            cases.append(
                {
                    "id": f"dev-{idx:03d}",
                    "dataset": "dev",
                    "type": qtype,
                    "state": state,
                    "question": question,
                    "expected": str(expected),
                    "criteria": criteria,
                }
            )

    return cases


def build_option_map(case):
    if case["type"] == "choice":
        return case["criteria"]
    if case["type"] == "noul":
        return {"yes": "yes", "no": "no"}
    return {str(i): label for i, label in enumerate(case["criteria"])}


def baseline_prompt(case):
    system = (
        "You are a routing classifier. Pick exactly one option id based on the state. "
        "Return compact JSON only: {\"id\":\"<option_id>\"}."
    )
    options = build_option_map(case)
    user = (
        f"State:\n{case['state']}\n\n"
        f"Question:\n{case['question']}\n\n"
        f"Options:\n{json.dumps(options, ensure_ascii=False, indent=2)}\n\n"
        "Answer with the option id."
    )
    return f"{system}\n\n{user}"


def tinyjev_question(case):
    question = {"type": case["type"], "instructions": case["question"]}
    if case["type"] in {"choice", "score"}:
        question["criteria"] = case["criteria"]
    return question


def extract_model_answer(case, answer):
    qtype = case["type"]
    if qtype == "noul":
        p = float(answer["noul"])
        model_answer = "yes" if p >= 0.5 else "no"
        confidence = max(p, 1.0 - p)
        correct = model_answer == case["expected"]
        return model_answer, confidence, correct

    if qtype == "choice":
        model_answer = str(answer["choice"])
        confidence = float(answer.get("confidence") or 0.0)
        correct = model_answer == case["expected"]
        return model_answer, confidence, correct

    score_value = float(answer["score"])
    model_answer = str(int(round(score_value)))
    confidence = float(answer.get("confidence") or 0.0)
    expected = int(case["expected"])
    correct = abs(score_value - expected) <= 1.0
    return model_answer, confidence, correct


def tool_call_tokens(case, count_tokens, min_confidence):
    tool_name = {
        "noul": "jev_yesno",
        "choice": "jev_choice",
        "score": "jev_score",
    }[case["type"]]
    args = {
        "state": case["state"],
        "question": case["question"],
        "min_confidence": min_confidence,
    }
    if case["type"] == "choice":
        args["options"] = case["criteria"]
    elif case["type"] == "score":
        args["levels"] = case["criteria"]

    payload = {"name": tool_name, "arguments": args}
    return count_tokens(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def load_cost_module(calc_path: Path):
    spec = importlib.util.spec_from_file_location("copilot_calc", calc_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tool_schema_tokens(count_tokens):
    from entschied.mcp import mcp

    async def _inner():
        tools = await mcp.list_tools()
        payload = [
            {"name": t.name, "description": t.description, "inputSchema": t.inputSchema}
            for t in tools
        ]
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return count_tokens(text)

    return asyncio.run(_inner())


def benchmark(args):
    count_tokens, tokenizer_name = load_token_counter()
    cases = load_cases(args.approach_script, args.dev_script, args.include_dev)
    schema_tok = tool_schema_tokens(count_tokens)

    rows = []
    token_rows = []
    latencies = []
    decided_correct = 0
    decided_total = 0
    deferred_total = 0

    client = httpx.Client(timeout=args.timeout)
    try:
        for case in cases:
            question = tinyjev_question(case)
            payload = {
                "state": case["state"],
                "min_confidence": args.min_confidence,
                "questions": {"result": question},
            }

            t0 = time.perf_counter()
            response = client.post(f"{args.url}/v1/systemone", json=payload)
            response.raise_for_status()
            body = response.json()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            answer = body["answers"]["result"]
            model_answer, confidence, correct = extract_model_answer(case, answer)
            decided = bool(answer.get("decided", confidence >= args.min_confidence))
            deferred = not decided

            if decided:
                decided_total += 1
                if correct:
                    decided_correct += 1
            else:
                deferred_total += 1

            row_ok = deferred or correct
            latencies.append(elapsed_ms)

            prompt = baseline_prompt(case)
            base_input = count_tokens(prompt)
            base_output_text = json.dumps({"id": case["expected"]}, ensure_ascii=False, separators=(",", ":"))
            base_output = count_tokens(base_output_text)

            result_text = json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
            mcp_result_input = count_tokens(result_text)
            mcp_call_output = tool_call_tokens(case, count_tokens, args.min_confidence)

            token_rows.append(
                {
                    "baseline_input": base_input,
                    "baseline_output": base_output,
                    "front_input": 0 if decided else base_input,
                    "front_output": 0 if decided else base_output,
                    "mcp_input": base_input + schema_tok + mcp_result_input,
                    "mcp_output": mcp_call_output + base_output,
                }
            )

            rows.append(
                {
                    "Dataset": case["dataset"],
                    "State": case["state"],
                    "Question": case["question"],
                    "Expected": case["expected"],
                    "Model answer": model_answer,
                    "Value": round(confidence, 4),
                    "ms": round(elapsed_ms, 2),
                    "decided": decided,
                    "ok": row_ok,
                }
            )
    finally:
        client.close()

    cost_mod = load_cost_module(args.calc_script)
    models = ["GPT-5 mini", "GPT-5.4", "Claude Sonnet 5", "Claude Opus 5"]
    totals = {}

    for model in models:
        baseline_cost = 0.0
        front_cost = 0.0
        mcp_cost = 0.0
        for row in token_rows:
            _, b_parts = cost_mod.cost(model, inp=row["baseline_input"], out=row["baseline_output"])
            _, f_parts = cost_mod.cost(model, inp=row["front_input"], out=row["front_output"])
            _, m_parts = cost_mod.cost(model, inp=row["mcp_input"], out=row["mcp_output"])
            baseline_cost += b_parts["input"] + b_parts["output"]
            front_cost += f_parts["input"] + f_parts["output"]
            mcp_cost += m_parts["input"] + m_parts["output"]

        scale = 1000.0 / len(token_rows)
        totals[model] = {
            "online_only": baseline_cost * scale,
            "tinyjev_front": front_cost * scale,
            "mcp_tool": mcp_cost * scale,
        }

    local_share = decided_total / len(rows) if rows else 0.0
    decided_accuracy = decided_correct / decided_total if decided_total else 0.0
    median_ms = statistics.median(latencies) if latencies else 0.0

    summary = {
        "cases": len(rows),
        "decided": decided_total,
        "deferred": deferred_total,
        "local_share": local_share,
        "decided_accuracy": decided_accuracy,
        "median_tinyjev_ms": median_ms,
        "tokenizer": tokenizer_name,
        "tool_schema_tokens": schema_tok,
        "online_latency_measured": False,
        "cost_per_1000_usd": totals,
    }

    args.rows_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)

    args.rows_out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_line = (
        f"local share {local_share:.1%}, decided accuracy {decided_accuracy:.1%}, median {median_ms:.0f} ms; "
        f"$/1k GPT-5 mini {totals['GPT-5 mini']['online_only']:.3f}/{totals['GPT-5 mini']['tinyjev_front']:.3f}/{totals['GPT-5 mini']['mcp_tool']:.3f}, "
        f"GPT-5.4 {totals['GPT-5.4']['online_only']:.3f}/{totals['GPT-5.4']['tinyjev_front']:.3f}/{totals['GPT-5.4']['mcp_tool']:.3f}, "
        f"Sonnet5 {totals['Claude Sonnet 5']['online_only']:.3f}/{totals['Claude Sonnet 5']['tinyjev_front']:.3f}/{totals['Claude Sonnet 5']['mcp_tool']:.3f}, "
        f"Opus5 {totals['Claude Opus 5']['online_only']:.3f}/{totals['Claude Opus 5']['tinyjev_front']:.3f}/{totals['Claude Opus 5']['mcp_tool']:.3f}"
    )

    subprocess.run(
        [
            args.python,
            str(args.renderer_script),
            str(args.rows_out),
            str(args.html_out),
            "--title",
            "TinyJev 0.6B token savings",
            "--summary",
            summary_line,
        ],
        check=True,
    )

    print(json.dumps(summary, indent=2))
    print(args.rows_out)
    print(args.summary_out)
    print(args.html_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8093")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--min-confidence", type=float, default=0.85)
    parser.add_argument(
        "--approach-script",
        type=Path,
        default=Path("C:/Users/halit/AppData/Local/hermes/cache/scratch/entschied_approach_bench.py"),
    )
    parser.add_argument(
        "--dev-script",
        type=Path,
        default=Path("C:/Users/halit/AppData/Local/hermes/cache/scratch/entschied_dev_bench.py"),
    )
    parser.add_argument("--include-dev", action="store_true", default=True)
    parser.add_argument(
        "--calc-script",
        type=Path,
        default=Path("C:/Users/halit/AppData/Local/hermes/skills/productivity/copilot-token-price-calculator/scripts/calc.py"),
    )
    parser.add_argument(
        "--renderer-script",
        type=Path,
        default=Path("C:/Users/halit/AppData/Local/hermes/skills/productivity/benchmark-html-report/scripts/render.py"),
    )
    parser.add_argument(
        "--rows-out",
        type=Path,
        default=Path("benchmarks/token-savings-0.6b-rows.json"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("benchmarks/token-savings-0.6b-summary.json"),
    )
    parser.add_argument(
        "--html-out",
        type=Path,
        default=Path("benchmarks/token-savings-0.6b.html"),
    )
    parser.add_argument(
        "--python",
        default=".venv/Scripts/python.exe",
    )

    args = parser.parse_args()
    benchmark(args)


if __name__ == "__main__":
    main()
