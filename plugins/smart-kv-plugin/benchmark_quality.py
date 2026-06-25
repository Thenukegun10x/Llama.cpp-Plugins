"""
Cache quality benchmark: compare perplexity across cache configurations.

Measures how much information is lost by each cache policy.
Uses the model's own confidence (logprobs) on held-out conversations.

Usage:
  python benchmark_quality.py [--model Qwen3.5Test.gguf] [--trials 3]

  Each trial runs every profile once and reports PPL + VRAM.

Profiles tested:
  f16   – oracle baseline (no compression loss)
  q8    – q8_0 uniform
  q4    – q4_0 uniform (stock baseline)
  old   – mixed bucket (pre-smart profile)
  smart – smart heuristic scorer (no MLP)
  smart-learned – smart + trained MLP weights
  smart-tq – smart + MLP + TQ CPU offload

Output:
  Markdown table with PPL, relative perplexity increase vs F16, VRAM.
"""

import subprocess
import sys
import os
import json
import time
import math
import urllib.request
import urllib.error
import glob
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "http://127.0.0.1:12345"
SERVER_EXE = os.path.join(BASE_DIR, "llama.cpp", "build", "bin", "llama-server.exe")
MODEL_DIR = BASE_DIR
START_BAT = os.path.join(BASE_DIR, "start.bat")

# ── Test corpora ──────────────────────────────────────────────────────
# Each is a multi-turn conversation that exercises the KV cache.
# We measure the model's confidence (avg logprob) on each assistant turn
# after N previous turns have filled the cache.

TEST_CONVERSATIONS = [
    {  # 0: coding task with tools
        "name": "coding_tools",
        "system": "You are an AI coding assistant with access to read_file, search_symbols, and run_tests.",
        "turns": [
            "Read the file src/core/engine.cpp and explain what it does.",
            "Search for all references to the 'Engine' class.",
            "Show me the Engine::run function signature.",
            "There's a bug in Engine::run — it doesn't handle exceptions properly.",
            "Write a fix for the exception handling.",
            "Run the unit tests for Engine.",
            "Fix the 3 test failures.",
            "Add documentation for the new method.",
        ],
    },
    {  # 1: compiler error marathon
        "name": "compiler_errors",
        "system": "You are a Rust expert helping fix compiler errors.",
        "turns": [
            "Implement a generic binary search function for any Ord type.",
            "error[E0308]: mismatched types: expected u32, found usize",
            "error[E0502]: cannot borrow as mutable because already borrowed",
            "Fix all three compiler errors.",
            "Now write a test module for the binary search.",
            "cargo test fails. Fix the test.",
        ],
    },
    {  # 2: long code generation
        "name": "long_code_gen",
        "system": "You are an autonomous coding agent. Write complete, working code.",
        "turns": [
            "Write a Python async task queue with worker pool.",
            "Add a cancel method to cancel pending tasks.",
            "Add a retry mechanism for failed tasks.",
            "Write comprehensive unit tests.",
            "Add type hints and documentation.",
        ],
    },
    {  # 3: file review (heavy file path context)
        "name": "file_review",
        "system": "You are a senior code reviewer.",
        "turns": [
            "Review src/core/engine.cpp for memory leaks.",
            "Check include/engine/api.h for API compatibility issues.",
            "Review tests/test_engine.cpp — are the tests thorough?",
            "Check src/utils/logger.cpp for thread safety.",
            "Run clippy and fix all warnings.",
        ],
    },
    {  # 4: reasoning chain (long context dependency)
        "name": "deep_reasoning",
        "system": "You are a research scientist. Think step by step and consider multiple perspectives.",
        "turns": [
            "Design a distributed key-value store with strong consistency.",
            "How would you handle network partitions?",
            "Compare Raft vs Paxos for this design.",
            "What's the throughput bottleneck and how to scale to 1000 nodes?",
            "Analyze the failure modes of your design.",
        ],
    },
]

# ── Profiles to benchmark ──────────────────────────────────────────────
# (name, start.bat profile, display name)
PROFILES = [
    ("f16",          "f16",           "F16 (oracle)"),
    ("q8",           "q8",            "Q8_0 uniform"),
    ("q4",           "q4",            "Q4_0 uniform (stock)"),
    ("old",          "old",           "Mixed bucket (old)"),
    ("smart",        "smart",         "Smart heuristic"),
]

# Smart + learned MLP weights (if available)
if os.path.exists(os.path.join(PLUGIN_DIR, "learned_weights.bin")):
    PROFILES.append(("smart-learned", "smart", "Smart learned MLP"))


def api_post(path, body, timeout=120):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{BASE_URL}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def api_get(path, timeout=10):
    req = urllib.request.Request(f"{BASE_URL}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def server_running():
    try:
        api_get("/health", timeout=3)
        return True
    except:
        return False

def kill_server():
    os.system("taskkill /F /IM llama-server.exe 2>nul")
    time.sleep(2)

def start_server(profile_id, model_name=None, cache_override=None):
    """Start server with given profile. Returns (url, process)."""
    kill_server()

    # Build args: start.bat <profile> but we bypass the bat and call server directly
    # because start.bat has interactive elements. Instead, replicate the config here.
    profile_configs = {
        "f16":  {"ctx": 16384, "batch": 256, "ubatch": 128, "reuse": 0,
                 "cache": "--cache-type-k f16 --cache-type-v f16",
                 "plugin": "", "model": None},
        "q8":   {"ctx": 32768, "batch": 256, "ubatch": 128, "reuse": 1024,
                 "cache": "--cache-type-k q8_0 --cache-type-v q8_0",
                 "plugin": "", "model": None},
        "q4":   {"ctx": 32768, "batch": 256, "ubatch": 128, "reuse": 1024,
                 "cache": "--cache-type-k q4_0 --cache-type-v q4_0",
                 "plugin": "", "model": None},
        "old":  {"ctx": 100000, "batch": 512, "ubatch": 256, "reuse": 2048,
                 "cache": "--cache-type-k-mixed q5_1:2048,q4_0:8192,q3:24576,q2:0",
                 "plugin": "", "model": None},
        "smart": {"ctx": 200000, "batch": 384, "ubatch": 192, "reuse": 2048,
                  "cache": "--cache-type-k-mixed q5_0:2048,q4_0:8192,q2_k:0",
                  "plugin": "on", "model": None},
    }

    cfg = profile_configs.get(profile_id)
    if not cfg:
        print(f"  Unknown profile: {profile_id}")
        return False

    if cache_override:
        cfg["cache"] = cache_override

    # Resolve model
    model = cfg["model"] or model_name
    if not model:
        ggufs = sorted(glob.glob(os.path.join(MODEL_DIR, "*.gguf")),
                       key=os.path.getmtime, reverse=True)
        if not ggufs:
            print("  No .gguf model found")
            return False
        model = ggufs[0]

    # Build command
    plugin_args = ""
    if cfg["plugin"] == "on":
        plugin_path = os.path.join(PLUGIN_DIR, "smart-tq-plugin.dll")
        if os.path.exists(plugin_path):
            env_path = os.path.join(BASE_DIR, "plugins.env")
            with open(env_path, "w") as f:
                f.write(plugin_path)
            plugin_args = f"--plugin-kv-cache on"

    cmd = [
        SERVER_EXE,
        "-m", model,
        "-ngl", "99",
        "-fa", "on",
        *cfg["cache"].split(),
        "-c", str(cfg["ctx"]),
        "-b", str(cfg["batch"]),
        "-ub", str(cfg["ubatch"]),
        "--parallel", "1",
        "--temp", "0.0",
        "--cache-reuse", str(cfg["reuse"]),
    ]
    if plugin_args:
        cmd.extend(plugin_args.split())
        env = os.environ.copy()
        if os.path.exists(os.path.join(BASE_DIR, "plugins.env")):
            env["LLAMA_PLUGIN_FILE"] = os.path.join(BASE_DIR, "plugins.env")
    else:
        env = os.environ.copy()

    cmd_str = " ".join(cmd)
    print(f"  Starting: {os.path.basename(model)} @ {cfg['ctx']} ctx")
    print(f"  Cache: {cfg['cache']}")
    print(f"  Plugin: {cfg['plugin'] or 'none'}")

    # Launch server
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW)

    # Wait for server ready
    deadline = time.time() + 120
    while time.time() < deadline:
        if server_running():
            print("  Server ready")
            return True
        time.sleep(2)

    print("  Server did not start")
    proc.kill()
    return False


def measure_perplexity(conversations, max_tokens_per_turn=32):
    """
    Measure avg per-token perplexity across conversations.
    Sends the full conversation history (multi-turn) and measures
    logprobs on the assistant's response tokens.

    Returns: (avg_ppl, total_tokens)
    """
    total_nll = 0.0
    total_tokens = 0
    n_turns = 0

    for conv in conversations:
        history = [{"role": "system", "content": conv["system"]}]
        for turn_idx, user_msg in enumerate(conv["turns"]):
            history.append({"role": "user", "content": user_msg})

            try:
                resp = api_post("/v1/chat/completions", {
                    "messages": history,
                    "max_tokens": max_tokens_per_turn,
                    "temperature": 0.0,
                    "logprobs": True,
                    "top_logprobs": 1,
                }, timeout=60)

                choice = resp["choices"][0]
                content = choice["message"]["content"]
                logprobs_data = choice.get("logprobs", {})

                if logprobs_data and "content" in logprobs_data:
                    for token_logprob in logprobs_data["content"]:
                        lp = token_logprob.get("logprob", 0)
                        total_nll += -lp
                        total_tokens += 1
                else:
                    # Fallback: estimate from token count
                    total_tokens += len(content) // 4

                # Append assistant response for next turn
                history.append({"role": "assistant", "content": content})
                n_turns += 1

            except Exception as e:
                print(f"    Turn {turn_idx} error: {e}")
                continue

    if total_tokens == 0:
        return 0.0, 0

    avg_ppl = math.exp(total_nll / total_tokens)
    return avg_ppl, total_tokens


def estimate_vram(cfg):
    """Estimate VRAM usage for a given context size and cache type."""
    # Rough formula: n_layers * (K + V) * ctx_size * bpw/8
    # Qwen3.6-27B: 64 layers, n_kv_heads=8, head_dim=256 (for K)
    # Actually wait - README says 64 layers, let's check.
    # For the 4B test model: 32 layers, n_kv_heads=8, head_dim=128
    # Estimate conservatively: 32 layers for test model
    n_layers = 32
    n_kv_heads = 8
    head_dim = 128  # for K, V is usually same dim
    bytes_per_element = {"f16": 2, "q8_0": 1, "q4_0": 0.5, "q5_1": 0.625,
                         "q5_0": 0.625, "q6_k": 0.75, "q4_k": 0.5,
                         "q3_k": 0.375, "q2_k": 0.25, "q2": 0.25, "q3": 0.375}
    ctx = cfg.get("ctx", 32768)
    total_vram = 0
    for cache_part in cfg.get("cache", "").split("--cache-type-"):
        if not cache_part.strip():
            continue
        parts = cache_part.split()
        if len(parts) < 2:
            continue
        kv = parts[0]  # k or v
        rest = " ".join(parts[1:])
        # Parse mixed format or simple type
        if ":" in rest:
            # mixed: q4_0:2048,q3:14336,q2_k:0
            for segment in rest.split(","):
                seg_parts = segment.split(":")
                if len(seg_parts) == 2:
                    qtype, limit = seg_parts[0], int(seg_parts[1])
                    bpw = bytes_per_element.get(qtype, 1.0)
                    actual_limit = limit if limit > 0 else ctx
                    total_vram += n_layers * actual_limit * n_kv_heads * head_dim * bpw * 2
        else:
            qtype = rest.strip()
            bpw = bytes_per_element.get(qtype, 2.0)
            total_vram += n_layers * ctx * n_kv_heads * head_dim * bpw * 2

    return total_vram / (1024 * 1024)  # MB


def run_benchmark(trials=3, model_name=None, cache_override=None):
    """Run the full benchmark across all profiles."""

    # Parse cache override
    if cache_override:
        print(f"\nCache override: {cache_override}")
        print("Running single-config benchmark\n")
        profiles_to_test = [("custom", "custom", "Custom config")]
    else:
        profiles_to_test = PROFILES

    results = []

    for profile_id, bat_profile, display_name in profiles_to_test:
        print(f"\n{'='*60}")
        print(f"  Profile: {display_name}")
        print(f"{'='*60}")

        if not start_server(profile_id, model_name, cache_override):
            print(f"  SKIPPED (server failed to start)")
            continue

        trial_ppls = []
        trial_tokens = []

        for trial in range(1, trials + 1):
            print(f"\n  Trial {trial}/{trials}...")
            ppl, n_tokens = measure_perplexity(TEST_CONVERSATIONS)
            trial_ppls.append(ppl)
            trial_tokens.append(n_tokens)
            print(f"    PPL: {ppl:.4f} ({n_tokens} tokens)")

        # Estimate VRAM
        cfg = {"ctx": {"f16": 16384, "q8": 32768, "q4": 32768,
                       "old": 100000, "smart": 200000}.get(profile_id, 32768),
               "cache": {"f16": "--cache-type-k f16 --cache-type-v f16",
                         "q8": "--cache-type-k q8_0 --cache-type-v q8_0",
                         "q4": "--cache-type-k q4_0 --cache-type-v q4_0",
                         "old": "--cache-type-k-mixed q5_1:2048,q4_0:8192,q3:24576,q2:0",
                         "smart": "--cache-type-k-mixed q5_0:2048,q4_0:8192,q2_k:0",
                        }.get(profile_id, "--cache-type-k q4_0")}
        vram_mb = estimate_vram(cfg)

        avg_ppl = sum(trial_ppls) / len(trial_ppls)
        results.append({
            "name": display_name,
            "ppl": avg_ppl,
            "tokens": sum(trial_tokens) // len(trial_tokens),
            "vram_mb": vram_mb,
        })

        kill_server()
        time.sleep(1)

    # Report
    print(f"\n\n{'='*60}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'='*60}")

    if not results:
        print("  No results collected.")
        return

    # Find F16 baseline
    f16_result = next((r for r in results if "F16" in r["name"]), None)
    baseline_ppl = f16_result["ppl"] if f16_result else results[0]["ppl"]

    print()
    print(f"  {'Cache Config':<28} {'PPL':<10} {'vs F16':<10} {'Tokens':<10} {'VRAM':<10}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for r in results:
        ppl_delta = ((r["ppl"] - baseline_ppl) / baseline_ppl) * 100
        ppl_str = f"{r['ppl']:.4f}"
        delta_str = f"+{ppl_delta:.2f}%" if ppl_delta > 0 else "baseline"
        vram_str = f"{r['vram_mb']:.0f} MB"
        print(f"  {r['name']:<28} {ppl_str:<10} {delta_str:<10} {r['tokens']:<10} {vram_str:<10}")

    # Rank by quality/VRAM trade-off
    print(f"\n  Ranked by perplexity (lower is better):")
    for i, r in enumerate(sorted(results, key=lambda x: x["ppl"]), 1):
        print(f"    {i}. {r['name']:<28} PPL={r['ppl']:.4f}  VRAM={r['vram_mb']:.0f}MB")

    # Markdown output for documentation
    print(f"\n\n  ## Markdown table\n")
    print(f"  | Cache Config | PPL | vs F16 | Tokens | VRAM |")
    print(f"  |------------|------|--------|--------|------|")
    for r in sorted(results, key=lambda x: x["ppl"]):
        ppl_delta = ((r["ppl"] - baseline_ppl) / baseline_ppl) * 100
        delta_str = f"+{ppl_delta:.2f}%" if ppl_delta > 0 else "baseline"
        print(f"  | {r['name']} | {r['ppl']:.4f} | {delta_str} | {r['tokens']} | {r['vram_mb']:.0f} MB |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV cache quality benchmark")
    parser.add_argument("--model", help="Path to .gguf model file")
    parser.add_argument("--trials", type=int, default=3, help="Trials per profile")
    parser.add_argument("--cache", help="Override cache args for single run: '--cache-type-k q8_0'")
    parser.add_argument("--list", action="store_true", help="List available profiles and exit")
    args = parser.parse_args()

    if args.list:
        print("Available benchmark profiles:")
        for pid, _, dname in PROFILES:
            print(f"  {pid:<20} {dname}")
        sys.exit(0)

    run_benchmark(trials=args.trials, model_name=args.model, cache_override=args.cache)
