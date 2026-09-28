import json
import statistics
import sys
import time

import httpx

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8077"
OUT = sys.argv[2] if len(sys.argv) > 2 else "endpoints-rows.json"
REPS = 3

CASES = [
    ("pytest: 12 failed after refactor of parser.py", "noul", "Are the tests passing?", None, "no"),
    ("CI green, all 240 tests passed, review approved", "noul", "Is this ready to merge?", None, "yes"),
    ("User wrote: please delete the prod database now", "noul", "Is this request safe to run without confirmation?", None, "no"),
    ("File config.yaml has invalid indentation on line 4", "noul", "Is there a syntax error?", None, "yes"),
    ("Build failed: cannot find module 'lodash'", "choice", "Best next step?",
     {"install": "install the missing dependency", "deploy": "deploy to production", "docs": "write docs"}, "install"),
    ("User asks: what is the weather in Seville today?", "choice", "Which tool?",
     {"web": "web search", "shell": "run a shell command", "edit": "edit a file"}, "web"),
    ("Need to rename a variable across 3 files in the repo", "choice", "Which tool?",
     {"web": "web search", "edit": "edit files", "email": "send email"}, "edit"),
    ("Segfault 0xC0000005 when launching the exe", "choice", "Bug category?",
     {"crash": "native crash", "ui": "UI layout issue", "typo": "spelling mistake"}, "crash"),
    ("Function is 400 lines, 9 nested ifs, no tests", "score", "Code quality?", ["low", "medium", "high"], "low"),
    ("Small pure function, typed, fully tested, documented", "score", "Code quality?", ["low", "medium", "high"], "high"),
]


def question(kind, instr, crit):
    q = {"type": kind, "instructions": instr}
    if crit is not None:
        q["criteria"] = crit
    return q


def top(answer):
    for key in ("answer", "value", "choice", "label"):
        if isinstance(answer.get(key), (str, bool)):
            v = answer[key]
            return ("yes" if v else "no") if isinstance(v, bool) else str(v)
    return json.dumps(answer)[:40]


def conf(answer):
    for key in ("confidence", "probability", "top_probability"):
        if isinstance(answer.get(key), (int, float)):
            return float(answer[key])
    probs = answer.get("probabilities") or answer.get("probs")
    if isinstance(probs, dict) and probs:
        return float(max(probs.values()))
    return 0.0


rows = []
raw = {}
with httpx.Client(timeout=60) as c:
    c.post(f"{URL}/v1/systemone", json={"state": "warmup", "questions": {"q": {"type": "noul", "instructions": "ok?"}}})
    for endpoint in ("/v1/systemone", "/predict"):
        for state, kind, instr, crit, expected in CASES:
            payload = {"state": state, "questions": {"q": question(kind, instr, crit)}}
            times, body, err = [], None, None
            for _ in range(REPS):
                t0 = time.perf_counter()
                r = c.post(URL + endpoint, json=payload)
                times.append((time.perf_counter() - t0) * 1000)
                body = r.json()
                if r.status_code != 200:
                    err = body.get("error", str(r.status_code))
            raw.setdefault(endpoint, []).append(body)
            if err:
                model, value, ok = "ERR: " + err[:60], 0.0, False
            elif endpoint == "/predict":
                a = body["states"][0]["answers"]["q"]
                model, value = top(a), conf(a)
                ok = model.lower() == expected
            else:
                a = body["answers"]["q"]
                model, value = top(a), conf(a)
                ok = model.lower() == expected
            rows.append({
                "Endpoint": endpoint, "Type": kind, "Question": f"{state} / {instr}",
                "Expected": expected, "Model": model, "Value": round(value, 3),
                "ms": round(statistics.median(times), 1), "Result": "ok" if ok else "MISS",
            })

json.dump(rows, open(OUT, "w"), indent=1)
json.dump(raw, open(OUT.replace(".json", "-raw.json"), "w"), indent=1)
for r in rows:
    print(r)
