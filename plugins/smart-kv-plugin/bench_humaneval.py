"""
HumanEval cache-quality benchmark for an already-running llama-server.

Run the server separately with either:
  start.bat f16
  start.bat smart

Then run this script once per profile and compare pass@1. The default stress
mode keeps a single growing conversation, but appends the canonical HumanEval
solution after each task so every cache profile sees the same prior history.
That avoids later prompts depending on a profile's earlier generated output.
"""

import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.request


BASE_URL = "http://127.0.0.1:12345"
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(PLUGIN_DIR, "..", ".."))


SYSTEM_PROMPT = (
    "You are completing Python HumanEval functions. "
    "Return only the Python implementation. "
    "Do not include markdown fences, explanations, tests, examples, or comments outside the function. "
    "Stop immediately after the function is complete."
)


def find_data_file(path=None):
    if path:
        return os.path.abspath(path)

    candidates = [
        os.path.join(PLUGIN_DIR, "HumanEval.jsonl.gz"),
        os.path.join(BASE_DIR, "HumanEval.jsonl.gz"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("HumanEval.jsonl.gz not found in plugin dir or repo root")


def load_problems(path=None):
    data_file = find_data_file(path)
    with gzip.open(data_file, "rt", encoding="utf-8") as f:
        problems = [json.loads(line) for line in f]
    return problems, data_file


def api_post(base_url, path, body, timeout):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def api_get(base_url, path, timeout):
    req = urllib.request.Request(f"{base_url}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def require_server(base_url):
    try:
        api_get(base_url, "/health", timeout=5)
    except Exception as e:
        raise RuntimeError(f"server is not reachable at {base_url}: {e}")


def strip_code_fences(text):
    text = text.strip()
    if "```" not in text:
        return text

    parts = text.split("```")
    if len(parts) < 3:
        return text.replace("```python", "").replace("```", "").strip()

    block = parts[1]
    lines = block.splitlines()
    if lines and lines[0].strip().lower() in {"python", "py"}:
        lines = lines[1:]
    return "\n".join(lines).strip()


def normalize_completion(prompt, completion, entry_point, repair_indent=False):
    completion = strip_code_fences(completion)

    marker = f"def {entry_point}("
    if marker in completion:
        start = completion.find(marker)
        return textwrap.dedent(completion[start:]).rstrip() + "\n"

    lines = completion.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)

    # Models sometimes put helper imports before a function body. Common
    # imports are already provided by the harness prefix; keeping them here can
    # break indentation when the rest of the answer is an indented body.
    while lines and lines[0].lstrip().startswith(("import ", "from ")):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)

    completion = "\n".join(lines).rstrip()

    if not completion.strip():
        return prompt.rstrip() + "\n"

    first = next((line for line in completion.splitlines() if line.strip()), "")
    top_level_prefixes = ("def ", "class ", "import ", "from ", "#")
    if first and not first.startswith((" ", "\t")) and not first.startswith(top_level_prefixes):
        if repair_indent:
            completion = "\n".join(
                f"    {line}" if line.strip() and not line.startswith((" ", "\t")) else line
                for line in completion.splitlines()
            )
        else:
            completion = textwrap.dedent(completion)
            completion = "\n".join(
                f"    {line}" if line.strip() else line
                for line in completion.splitlines()
            )

    completion = completion.rstrip() + "\n"

    if prompt.endswith("\n"):
        return prompt + completion
    return prompt + "\n" + completion


def build_test_program(problem, completion, repair_indent=False):
    prompt = problem["prompt"]
    entry_point = problem["entry_point"]
    solution = normalize_completion(prompt, completion, entry_point, repair_indent=repair_indent)
    prefix = (
        "from typing import *\n"
        "from collections import *\n"
        "import math\n"
        "import re\n"
        "import itertools\n"
        "import collections\n"
        "import functools\n"
        "import heapq\n"
        "import bisect\n\n"
    )
    return prefix + solution + "\n\n" + problem["test"] + f"\n\ncheck({entry_point})\n"


def run_test_once(problem, completion, timeout, repair_indent=False):
    program = build_test_program(problem, completion, repair_indent=repair_indent)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(program)
        path = f.name

    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, ""
        err = (result.stderr or result.stdout or "failed").strip()
        return False, err.splitlines()[-1] if err else "failed"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def run_test(problem, completion, timeout):
    ok, error = run_test_once(problem, completion, timeout, repair_indent=False)
    if ok:
        return ok, error
    if "IndentationError" in error:
        repaired_ok, repaired_error = run_test_once(problem, completion, timeout, repair_indent=True)
        if repaired_ok:
            return True, ""
        return False, repaired_error
    return ok, error


def estimate_tokens(messages):
    chars = sum(len(m.get("content", "")) for m in messages)
    return max(1, chars // 4)


def chat_completion(base_url, messages, max_tokens, temperature, seed, timeout):
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "stop": ["\nif __name__", "\n# Test", "\nassert ", "\nprint("],
    }
    return api_post(base_url, "/v1/chat/completions", body, timeout=timeout)


def canonical_history_message(problem):
    return {
        "role": "assistant",
        "content": strip_code_fences(problem.get("canonical_solution", "")).strip(),
    }


def generated_history_message(raw_completion):
    return {
        "role": "assistant",
        "content": strip_code_fences(raw_completion).strip(),
    }


def run_benchmark(args):
    require_server(args.url)
    problems, data_file = load_problems(args.data)
    if args.limit > 0:
        problems = problems[: args.limit]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    passed = 0
    failed = 0
    truncated = 0
    api_failed = 0
    context_overflow = 0
    records = []
    started = time.time()
    out_fp = open(args.out, "w", encoding="utf-8") if args.out else None

    def save_record(record):
        records.append(record)
        if out_fp:
            out_fp.write(json.dumps(record) + "\n")
            out_fp.flush()

    print(f"HumanEval cache benchmark")
    print(f"  profile label: {args.profile}")
    print(f"  server:        {args.url}")
    print(f"  data:          {data_file}")
    print(f"  mode:          {args.mode}")
    print(f"  history:       {args.history}")
    print(f"  problems:      {len(problems)}")
    print(f"  ctx hint:      {args.ctx}")
    print()

    try:
        for index, problem in enumerate(problems, 1):
            task_id = problem["task_id"]

            if args.mode == "isolated":
                request_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            else:
                request_messages = list(messages)

            request_messages.append({"role": "user", "content": problem["prompt"]})
            est_tokens = estimate_tokens(request_messages)
            pressure = ""
            if args.ctx and est_tokens >= args.ctx:
                pressure = " OVER_CTX"
            elif args.ctx and est_tokens >= int(args.ctx * 0.85):
                pressure = " NEAR_CTX"

            print(
                f"[{index:03d}/{len(problems):03d}] {task_id:<18} "
                f"est_prompt={est_tokens:>6} tok{pressure} ... ",
                end="",
                flush=True,
            )

            try:
                response = chat_completion(
                    args.url,
                    request_messages,
                    args.max_tokens,
                    args.temperature,
                    args.seed,
                    args.request_timeout,
                )
                choice = response["choices"][0]
                raw_completion = choice["message"].get("content", "")
                finish_reason = choice.get("finish_reason", "")
            except urllib.error.HTTPError as e:
                api_failed += 1
                failed += 1
                body = e.read().decode("utf-8", errors="replace")
                msg = f"HTTP {e.code}: {body or e.reason}"
                is_context = e.code == 400 and ("context" in msg.lower() or "tokens" in msg.lower())
                if is_context:
                    context_overflow += 1
                    print(f"CONTEXT_OVERFLOW {msg[:160]}")
                else:
                    print(f"API_FAIL {msg[:160]}")
                save_record({
                    "profile": args.profile,
                    "task_id": task_id,
                    "ok": False,
                    "status": "context_overflow" if is_context else "api_fail",
                    "error": f"api: {msg}",
                    "est_prompt_tokens": est_tokens,
                })
                if is_context and args.stop_on_context_overflow:
                    break
                continue
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                api_failed += 1
                failed += 1
                print(f"API_FAIL {e}")
                save_record({
                    "profile": args.profile,
                    "task_id": task_id,
                    "ok": False,
                    "status": "api_fail",
                    "error": f"api: {e}",
                    "est_prompt_tokens": est_tokens,
                })
                continue

            ok, error = run_test(problem, raw_completion, args.test_timeout)
            if finish_reason == "length":
                truncated += 1

            if ok:
                passed += 1
                result = "PASS"
            else:
                failed += 1
                result = "FAIL"

            suffix = f" ({error[:90]})" if error else ""
            if finish_reason == "length":
                suffix += " [TRUNC]"
            print(result + suffix)

            save_record({
                "profile": args.profile,
                "task_id": task_id,
                "ok": ok,
                "status": "pass" if ok else "fail",
                "error": error,
                "finish_reason": finish_reason,
                "est_prompt_tokens": est_tokens,
                "completion_chars": len(raw_completion),
                "completion_preview": raw_completion[:1000],
            })

            if args.mode == "stress":
                messages.append({"role": "user", "content": problem["prompt"]})
                if args.history == "canonical":
                    messages.append(canonical_history_message(problem))
                elif args.history == "generated":
                    messages.append(generated_history_message(raw_completion))
    finally:
        if out_fp:
            out_fp.close()

    elapsed = time.time() - started
    total = len(problems)
    attempted = len(records)
    quality_failed = max(0, failed - api_failed)
    scored = passed + quality_failed
    pass_rate = (100.0 * passed / scored) if scored else 0.0

    print()
    print("Result")
    print(f"  profile:       {args.profile}")
    print(f"  pass@1 scored: {passed}/{scored} = {pass_rate:.1f}%")
    print(f"  attempted:     {attempted}/{total}")
    print(f"  failed:        {quality_failed}")
    print(f"  truncated:     {truncated}")
    print(f"  api failed:    {api_failed}")
    print(f"  ctx overflow:  {context_overflow}")
    print(f"  elapsed:       {elapsed:.1f}s")
    print(f"  final history: {estimate_tokens(messages)} est tokens")

    if args.out:
        print(f"  wrote:         {args.out}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Valid HumanEval cache quality/stress benchmark for a running llama-server."
    )
    parser.add_argument("--url", default=BASE_URL, help="llama-server base URL")
    parser.add_argument("--profile", default="manual", help="label stored in output, e.g. f16 or smart")
    parser.add_argument("--data", help="path to HumanEval.jsonl.gz")
    parser.add_argument("--limit", type=int, default=60, help="number of problems; 0 means all")
    parser.add_argument("--ctx", type=int, default=32768, help="context size hint for pressure warnings")
    parser.add_argument("--mode", choices=["stress", "isolated"], default="stress")
    parser.add_argument(
        "--history",
        choices=["canonical", "generated", "none"],
        default="canonical",
        help="assistant messages appended in stress mode; canonical is fairest across profiles",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--test-timeout", type=int, default=10)
    parser.add_argument("--out", help="write per-task JSONL results")
    parser.add_argument(
        "--no-stop-on-context-overflow",
        dest="stop_on_context_overflow",
        action="store_false",
        help="continue after HTTP 400 context overflow instead of stopping the run",
    )
    parser.set_defaults(stop_on_context_overflow=True)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run_benchmark(parse_args())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
