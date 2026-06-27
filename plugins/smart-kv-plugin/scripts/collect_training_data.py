"""
Collect higher-quality Smart KV training data from a running llama-server.

Recommended server preset:
  start.bat test

That preset enables the smart KV plugin in training mode, disables old learned
weights during collection, uses heuristic teacher labels, and exports datasets
into this plugin directory.
"""

import argparse
import glob
import json
import os
import struct
import sys
import time
import urllib.request


BASE_URL = "http://127.0.0.1:12345"
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(PLUGIN_DIR, "..", ".."))
TRAINER = os.path.join(ROOT_DIR, "train_learned_score.py")


def api_post(base_url, path, body, timeout=180):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def api_get(base_url, path, timeout=10):
    req = urllib.request.Request(f"{base_url}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def estimate_tokens(messages):
    return max(1, sum(len(m.get("content", "")) for m in messages) // 4)


def send_chat(base_url, messages, max_tokens, temperature, seed):
    body = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    data = api_post(base_url, "/v1/chat/completions", body)
    text = data["choices"][0]["message"].get("content", "")
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", estimate_tokens(messages))
    return text, prompt_tokens


def code_blob(round_id):
    return f"""
// src/core/cache_manager_{round_id}.cpp
class CacheManager {{
public:
    bool insert(std::string key, std::vector<float> value);
    std::optional<std::vector<float>> lookup(const std::string& key) const;
    void compact(size_t target_bytes);
private:
    std::unordered_map<std::string, Entry> entries_;
    std::mutex mu_;
}};

bool CacheManager::insert(std::string key, std::vector<float> value) {{
    if (key.empty()) return false;
    std::lock_guard<std::mutex> lock(mu_);
    entries_[key] = Entry{{std::move(value), Clock::now(), {round_id}}};
    return true;
}}
""".strip()


def log_blob(round_id):
    lines = []
    for i in range(34):
        level = ["INFO", "DEBUG", "WARN", "ERROR", "TRACE"][i % 5]
        lines.append(
            f"[2026-06-25 12:{round_id:02d}:{i:02d}] [{level}] "
            f"worker={i % 7} shard=kv-{(i + round_id) % 4} event=cache_probe "
            f"latency_ms={17 + i * 3} bytes={4096 + i * 513}"
        )
    return "\n".join(lines)


def tool_schema_blob(round_id):
    schema = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": f"read_cache_trace_{round_id}",
                    "description": "Read a cache trace and return chunk reuse statistics.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "include_kv_norms": {"type": "boolean"},
                            "window": {"type": "integer", "minimum": 1},
                        },
                        "required": ["path"],
                    },
                },
            }
        ]
    }
    return json.dumps(schema, indent=2)


def make_sessions():
    return [
        {
            "name": "coding_debug",
            "max_tokens": 220,
            "system": "You are a senior C++ performance engineer. Preserve file names, symbols, and error messages exactly.",
            "turns": lambda r: [
                f"Memorize this implementation detail for later:\n{code_blob(r)}",
                "Find the lifetime bug in CacheManager::insert and explain the risk briefly.",
                "Compiler output: error: use of deleted function 'std::mutex::mutex(const std::mutex&)' in src/core/cache_manager.cpp:41. Fix the design.",
                "Now recall the exact private field names from the first snippet and propose a minimal patch.",
                "Write a focused regression test path tests/test_cache_manager.cpp that would catch this bug.",
            ],
        },
        {
            "name": "delayed_recall",
            "max_tokens": 160,
            "system": "You are a precise assistant. Track named facts and recall them later without inventing details.",
            "turns": lambda r: [
                f"Anchor facts: project=Orchid-{r}; owner=Marin; hot file=src/kv/reuse_map_{r}.rs; threshold={0.71 + r * 0.01:.2f}; failing test=test_reuse_after_wrap.",
                "Explain how a cache reuse map should behave under wraparound pressure.",
                "Summarize the difference between hot prefix tokens and cold generated prose.",
                "Without scrolling back, what was the project name, hot file, threshold, and failing test?",
                "Use those exact anchor facts in a short incident report.",
            ],
        },
        {
            "name": "logs_noise",
            "max_tokens": 120,
            "system": "You are an SRE. Separate important error facts from repetitive logs.",
            "turns": lambda r: [
                f"Analyze this noisy log block and identify only the important lines:\n{log_blob(r)}",
                "Which two signals should stay in high-quality cache and which repeated lines are safe to demote?",
                "A later alert says shard kv-2 regressed again. What earlier evidence matters?",
                "Write a concise remediation checklist using exact shard names when known.",
            ],
        },
        {
            "name": "tool_schema",
            "max_tokens": 180,
            "system": "You are an agent runtime designer. Tool schemas and argument names are important.",
            "turns": lambda r: [
                f"Register this tool schema and remember the required fields:\n{tool_schema_blob(r)}",
                "Generate a valid tool call JSON for reading traces/session.bin with KV norms enabled.",
                "Now modify the call to use a window of 128 and preserve the function name exactly.",
                "What field was required by the schema, and what optional boolean controlled KV norms?",
            ],
        },
        {
            "name": "mixed_commands",
            "max_tokens": 170,
            "system": "You are a build engineer. Commands, flags, and paths are high-value context.",
            "turns": lambda r: [
                f"Run sequence: git status --short; python -m pytest tests/test_kv_{r}.py -q; cmake --build build --target smart-tq-plugin",
                "pytest output: FAILED tests/test_kv_wrap.py::test_retrieve_after_demote - AssertionError: decoded K differs at slot 8192",
                "Suggest the smallest code area to inspect based on that failure.",
                "Repeat the exact failing pytest node and the suspicious slot number.",
            ],
        },
        {
            "name": "long_reasoning",
            "max_tokens": 240,
            "system": "You are a systems researcher. Keep the important constraints from earlier turns available.",
            "turns": lambda r: [
                f"Design constraints: context={32768 + r * 1024}, gpu_budget=14GB, ram_budget=64GB, target=HumanEval quality within 1 percent of F16.",
                "Compare FIFO, LRU, attention-based, and learned KV demotion for this target.",
                "Introduce a counterexample where naive recency hurts code-generation quality.",
                "Now use the original numeric constraints to choose a safe demotion policy.",
            ],
        },
        # ── Ping-pong scenario: force demote → promote → demote cycle ──
        {
            "name": "ping_pong_recall",
            "max_tokens": 200,
            "system": "You are a long-session chatbot. Many turns reference a single critical fact block written early. I will switch between recalling it and flooding the context with unrelated filler. You must preserve the fact block exactly.",
            "turns": lambda r: [
                f"CRITICAL: Store these facts verbatim. project=PingPong-{r}, threshold={0.35 + r * 0.05:.2f}, "
                f"hot_path=src/kv/ping_pong_{r}.rs, fail_test=test_promote_demote_cycle_{r}.",
                f"Filler: write a 200-word summary of Rust's borrow checker history, focusing on NLL adoption.",
                f"Recall the exact project name, threshold, hot path, and failing test from the critical fact block.",
                f"More filler: write 200 words comparing SQLite and DuckDB row-group storage formats.",
                f"Again: what was the project, threshold, hot path, and failing test? Answer with exact values only.",
            ],
        },
        # ── Alternating topics: exercises re-access of aged-out chunks ──
        {
            "name": "alternating_facts",
            "max_tokens": 220,
            "system": "You track two independent fact sets (SET-A and SET-B). I will alternate queries between them. When asked about one set, the other grows cold.",
            "turns": lambda r: [
                f"SET-A details: queue_policy=LRU_{r}, eviction_batch={128 + r * 8}, "
                f"warmup_tokens={2048 + r * 256}, demote_on_overflow=true.",
                f"SET-B details: queue_policy=CLOCK_{r}, sweep_interval={500 + r * 50}, "
                f"max_scan={64 + r * 4}, demote_on_age=true, max_age_steps={10000 + r * 100}.",
                f"From SET-A: what is the eviction_batch and demote_on_overflow?",
                f"From SET-B: what is the sweep_interval and max_scan?",
                f"From SET-A again: what is the warmup_tokens and eviction_batch?",
                f"From SET-B again: what is the max_age_steps and demote_on_age?",
            ],
        },
        # ── Rapid context switch: high-frequency ping-pong ──
        {
            "name": "rapid_switch",
            "max_tokens": 160,
            "system": "You respond with short, precise answers. The conversation history contains multiple topics that I will rapidly switch between.",
            "turns": lambda r: [
                f"Topic-{r}A: anchor=alpha_{r}, value=42.",
                f"Topic-{r}B: anchor=beta_{r}, value=99.",
                f"Topic-{r}C: anchor=gamma_{r}, value=7.",
                f"Topic-{r}A anchor?",
                f"Topic-{r}B anchor?",
                f"Topic-{r}C anchor?",
                f"Topic-{r}A again?",
                f"Topic-{r}B again?",
                f"Topic-{r}C again?",
            ],
        },
    ]


def exported_files(output_dir):
    patterns = [
        os.path.join(output_dir, "training_data_*.bin"),
        os.path.join(output_dir, "session_*.bin"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    return sorted(set(files), key=os.path.getmtime)


def remove_old_exports(output_dir):
    for path in exported_files(output_dir):
        try:
            os.remove(path)
            print(f"  removed old export: {os.path.basename(path)}")
        except OSError as e:
            print(f"  warning: could not remove {path}: {e}")


def verify_binary(path):
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            header = f.read(8)
            if len(header) != 8:
                return False, "too small"
            n_samples_f, n_features_f = struct.unpack("ff", header)
            n_samples = int(n_samples_f)
            n_features = int(n_features_f)
            if n_features not in (21, 23):
                return False, f"expected 21 or 23 features, got {n_features}"
            expected = 8 + n_samples * n_features * 4 + n_samples * 4 * 3
            if size != expected:
                return False, f"size mismatch {size} != {expected}"
            raw_x = f.read(n_samples * n_features * 4)
            raw_q = f.read(n_samples * 4)
            raw_r = f.read(n_samples * 4)
            raw_m = f.read(n_samples * 4)
        q = struct.unpack(f"{n_samples}f", raw_q) if n_samples else []
        r = struct.unpack(f"{n_samples}f", raw_r) if n_samples else []
        m = struct.unpack(f"{n_samples}f", raw_m) if n_samples else []
        ram_pos = sum(1 for x in r if x > 0.5)
        q_min = min(q) if q else 0.0
        q_max = max(q) if q else 0.0
        mem_mean = sum(m) / len(m) if m else 0.0
        warnings = []
        if n_samples < 1000:
            warnings.append("low sample count")
        if ram_pos == 0:
            warnings.append("no positive RAM labels")
        elif ram_pos < max(5, n_samples // 200):
            warnings.append("very few RAM positives")
        if q_max - q_min < 0.05:
            warnings.append("quant labels have low spread")
        summary = (
            f"{n_samples} samples, q=[{q_min:.3f},{q_max:.3f}], "
            f"ram_pos={ram_pos}/{n_samples}, mem_mean={mem_mean:.3f}"
        )
        if warnings:
            summary += " WARNING: " + ", ".join(warnings)
        return True, summary
    except Exception as e:
        return False, str(e)


def run_collection(args):
    print("SMART KV training data collector")
    print(f"  server:     {args.url}")
    print(f"  output dir: {args.output_dir}")
    print(f"  rounds:     {args.rounds}")
    print(f"  temp:       {args.temperature}")
    print()

    try:
        api_get(args.url, "/health", timeout=5)
    except Exception as e:
        raise RuntimeError(f"server is not reachable at {args.url}: {e}")

    try:
        models = api_get(args.url, "/v1/models", timeout=5)
        model_name = models.get("data", [{}])[0].get("id", "unknown")
        print(f"  model:      {model_name}")
    except Exception:
        print("  model:      unknown")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.clean:
        remove_old_exports(args.output_dir)

    sessions = make_sessions()
    histories = {
        s["name"]: [{"role": "system", "content": s["system"]}]
        for s in sessions
    }

    total_prompt_tokens = 0
    total_requests = 0
    started = time.time()

    for round_id in range(1, args.rounds + 1):
        print(f"\nRound {round_id}/{args.rounds}")
        for session in sessions:
            name = session["name"]
            turns = session["turns"](round_id)
            history = histories[name]
            for turn_idx, user_text in enumerate(turns, 1):
                history.append({"role": "user", "content": user_text})
                est = estimate_tokens(history)
                print(f"  {name:<15} turn {turn_idx}/{len(turns)} prompt~{est:>5} ... ", end="", flush=True)
                try:
                    reply, prompt_tokens = send_chat(
                        args.url,
                        history,
                        session["max_tokens"],
                        args.temperature,
                        args.seed + round_id + turn_idx,
                    )
                except Exception as e:
                    print(f"ERROR {e}")
                    history.pop()
                    continue
                history.append({"role": "assistant", "content": reply})
                total_prompt_tokens += prompt_tokens
                total_requests += 1
                print(f"ok ({prompt_tokens} prompt tokens)")
                time.sleep(args.pause)

        files = exported_files(args.output_dir)
        if files:
            newest = files[-1]
            ok, summary = verify_binary(newest)
            status = "verified" if ok else "invalid"
            print(f"  newest export: {os.path.basename(newest)} ({status}: {summary})")
        else:
            print("  newest export: none yet")

    elapsed = time.time() - started
    print("\nCollection complete")
    print(f"  requests:            {total_requests}")
    print(f"  summed prompt tokens: {total_prompt_tokens:,}")
    print(f"  elapsed:             {elapsed:.1f}s")

    files = exported_files(args.output_dir)
    if not files:
        print("\nNo export files found yet. Keep the server running until an auto-export threshold is reached, or stop the server to trigger session export.")
        return

    print("\nExported datasets")
    best = None
    best_size = -1
    for path in files:
        ok, summary = verify_binary(path)
        size = os.path.getsize(path)
        print(f"  {os.path.basename(path):<28} {size:>9} bytes  {summary}")
        if ok and size > best_size:
            best = path
            best_size = size

    if best:
        print("\nTrain with:")
        print(f"  python \"{TRAINER}\" \"{best}\" \"{args.output_dir}\"")
        print("\nThen restart normal inference with:")
        print("  start.bat smart")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect Smart KV learned-scorer training data.")
    parser.add_argument("rounds", nargs="?", type=int, default=4, help="collection rounds; default 4")
    parser.add_argument("--url", default=BASE_URL)
    parser.add_argument("--output-dir", default=PLUGIN_DIR)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--pause", type=float, default=0.05)
    parser.add_argument("--clean", action="store_true", help="remove old training_data/session exports first")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run_collection(parse_args())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
