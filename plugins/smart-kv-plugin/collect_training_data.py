"""
Multi-trial data collector for learned scorer. Runs each scenario N times.

Produces clean training data by simulating realistic agentic coding sessions
with proper multi-turn conversation context preserved.

Usage:
  start.bat test                     # with Qwen 3.5 4B
  python collect_training_data.py 5  # 5 trials → ~2000+ samples
  python train_learned_score.py training_data_4000.bin
"""

import urllib.request
import urllib.error
import sys
import time
import json
import os
import glob

BASE_URL = "http://127.0.0.1:12345"
N_TRIALS = 5 if len(sys.argv) < 2 else int(sys.argv[1])
MODEL_NAME = None  # auto-detected from /v1/models


def _api_post(path, body, timeout=120):
    """POST to the V1 API, return parsed JSON."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{BASE_URL}{path}", data=data,
        headers={"Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def _api_get(path, timeout=10):
    """GET from the V1 API, return parsed JSON."""
    req = urllib.request.Request(f"{BASE_URL}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def send_chat(messages, max_tokens=512):
    """Send a chat completion, returns (reply_text, prompt_tokens)."""
    global MODEL_NAME
    data = _api_post("/v1/chat/completions", {
        "model": MODEL_NAME or "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    })
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    pt = usage.get("prompt_tokens", len(json.dumps(messages)) // 4)
    return text, pt


def run_scenario(scenario):
    """
    Run one scenario as a proper multi-turn conversation.
    Builds history incrementally: each user message is followed by
    an assistant reply appended to history for the next turn.
    """
    original = scenario["gen"]()
    history = []
    total_tokens = 0

    for msg in original:
        history.append(msg)

        if msg["role"] == "system":
            continue

        # Send entire conversation history (system + prior turns + this msg)
        try:
            reply, ptokens = send_chat(history, max_tokens=scenario.get("max_tokens", 256))
            total_tokens += ptokens
            # Append assistant reply so next turn sees it
            history.append({"role": "assistant", "content": reply})
        except Exception as e:
            print(f"      error: {e}")
            break

        time.sleep(0.2)

    return total_tokens


# ── Scenarios ─────────────────────────────────────────────────────────
# Each scenario simulates a realistic agentic coding session with 10-20
# user/assistant exchanges to build up a meaningful KV cache.

def _sys_simple(msg="You are a helpful AI coding assistant."):
    return lambda: [{"role": "system", "content": msg}]

SCENARIOS = [

    # ── 1. Coding session with tools ────────────────────────────────
    {
        "name": "coding_session",
        "max_tokens": 256,
        "gen": lambda: _sys_simple("You are an AI coding assistant with access to read_file, search_symbols, and run_tests.")() + [
            {"role": "user", "content": "Read the file src/core/engine.cpp and explain what it does."},
            {"role": "user", "content": "Search for all references to the 'Engine' class."},
            {"role": "user", "content": "Show me the Engine::run function."},
            {"role": "user", "content": "There's a bug in Engine::run — it doesn't handle exceptions."},
            {"role": "user", "content": "Write a fix for the exception handling."},
            {"role": "user", "content": "Run the unit tests for Engine."},
            {"role": "user", "content": "Test results: 3 failed, 12 passed. Fix the failures."},
            {"role": "user", "content": "Now add a new method Engine::stop to gracefully shut down."},
            {"role": "user", "content": "Add documentation for the new method."},
            {"role": "user", "content": "Run all tests again to verify."},
        ],
    },

    # ── 2. Compiler error marathon ──────────────────────────────────
    {
        "name": "compiler_errors",
        "max_tokens": 192,
        "gen": lambda: _sys_simple("You are a Rust expert.")() + [
            {"role": "user", "content": "Implement a generic binary search function for any Ord type."},
            {"role": "user", "content": "error[E0308]: mismatched types: expected u32, found usize"},
            {"role": "user", "content": "error[E0502]: cannot borrow as mutable because already borrowed"},
            {"role": "user", "content": "error[E0382]: use of moved value"},
            {"role": "user", "content": "Fix all three compiler errors."},
            {"role": "user", "content": "warning: unused variable 'idx' — fix this too."},
            {"role": "user", "content": "Now write a test module for the binary search."},
            {"role": "user", "content": "cargo test --test binary_search — fails. Fix."},
        ],
    },

    # ── 3. Agentic loop: write → test → fix → verify ──────────────
    {
        "name": "agentic_loop",
        "max_tokens": 256,
        "gen": lambda: _sys_simple("You are an autonomous coding agent.")() + [
            {"role": "user", "content": "Write a Python async task queue with worker pool."},
            {"role": "user", "content": "$ python -m pytest tests/test_queue.py"},
            {"role": "user", "content": "FAIL: test_queue_submit — RuntimeError: queue already stopped"},
            {"role": "user", "content": "Fix the RuntimeError in the submit method."},
            {"role": "user", "content": "$ python -m pytest tests/test_queue.py -v"},
            {"role": "user", "content": "FAIL: test_queue_parallel_submit — deadlock detected"},
            {"role": "user", "content": "Fix the deadlock in the worker pool."},
            {"role": "user", "content": "$ python -m pytest tests/test_queue.py -v"},
            {"role": "user", "content": "All passed. Now add a cancel method."},
            {"role": "user", "content": "Write docs for the cancel method."},
        ],
    },

    # ── 4. File navigation + review ─────────────────────────────────
    {
        "name": "file_review",
        "max_tokens": 256,
        "gen": lambda: _sys_simple("You are a senior code reviewer.")() + [
            {"role": "user", "content": "Review src/core/engine.cpp for memory leaks."},
            {"role": "user", "content": "Check include/engine/api.h for API compatibility issues."},
            {"role": "user", "content": "Review tests/test_engine.cpp — are the tests thorough enough?"},
            {"role": "user", "content": "Check src/utils/logger.cpp for thread safety."},
            {"role": "user", "content": "Review the Cargo.toml for dependency issues."},
            {"role": "user", "content": "Run clippy on the entire project."},
            {"role": "user", "content": "Clippy warnings: 3 style issues, 1 correctness issue."},
            {"role": "user", "content": "Fix all clippy warnings."},
            {"role": "user", "content": "Re-run clippy to verify."},
        ],
    },

    # ── 5. Boilerplate + logs (cold data) ──────────────────────────
    {
        "name": "log_analysis",
        "max_tokens": 128,
        "gen": lambda: _sys_simple("You are a system administrator.")() + [
            {"role": "user", "content":
                "[2024-01-15 10:23:45] [INFO] Server starting on port 8080\n"
                "[2024-01-15 10:23:46] [DEBUG] Loading config from /etc/server.yaml\n"
                "[2024-01-15 10:23:47] [WARN] Deprecated field 'max_users' in config\n"
                "[2024-01-15 10:23:48] [INFO] Connected to database postgres://localhost:5432\n"
                "[2024-01-15 10:23:50] [ERROR] Connection pool exhausted: 100/100 connections\n"
                "[2024-01-15 10:23:51] [CRITICAL] Service unavailable — retry in 30s\n"
                "[2024-01-15 10:24:00] [INFO] Server shutting down\n"
                "What caused the crash?"},
            {"role": "user", "content": "How can we prevent this in the future?"},
            {"role": "user", "content": "Dump the last 100 log entries from /var/log/app.log"},
            {"role": "user", "content": "Filter for all ERROR and CRITICAL entries."},
        ],
    },

    # ── 6. Long multi-topic conversation (recency decay test) ──────
    {
        "name": "long_chat",
        "max_tokens": 128,
        "gen": lambda: [
            {"role": "system", "content": "You are knowledgeable about many topics."},
        ] + [{"role": "user", "content": p} for p in [
            "Explain how transformers work.",
            "What is the difference between L1 and L2 regularization?",
            "Write a haiku about machine learning.",
            "Explain the CAP theorem.",
            "What are the advantages of Rust over C++?",
            "Describe the OSI model layers.",
            "What is the difference between TCP and UDP?",
            "Explain how garbage collection works.",
            "What is the halting problem?",
            "Describe the difference between SVM and neural networks.",
            "Explain what a blockchain actually is.",
            "What is the P vs NP problem?",
            "Describe how a database index works.",
            "What is the difference between HTTP/2 and HTTP/3?",
        ]],
    },

    # ── 7. Commands + build output ──────────────────────────────────
    {
        "name": "build_cycle",
        "max_tokens": 256,
        "gen": lambda: _sys_simple("You are a build engineer.")() + [
            {"role": "user", "content": "./configure --prefix=/usr/local --enable-optimizations"},
            {"role": "user", "content": "make -j$(nproc) 2>&1 | tail -50"},
            {"role": "user", "content": "error: 'Tensor' was not declared in this scope"},
            {"role": "user", "content": "Fix the Tensor declaration in src/tensor.hpp"},
            {"role": "user", "content": "make -j$(nproc) 2>&1 | tail -50"},
            {"role": "user", "content": "collect2: error: ld returned 1 exit status — undefined reference to 'tensor_gemm'"},
            {"role": "user", "content": "Fix the linker error in CMakeLists.txt"},
            {"role": "user", "content": "cmake --build build --target all"},
            {"role": "user", "content": "Build succeeded! Now run the benchmark suite."},
            {"role": "user", "content": "Benchmark results: 2.3x speedup over baseline."},
        ],
    },

    # ── 8. Long system prompt + deep reasoning ─────────────────────
    {
        "name": "deep_reasoning",
        "max_tokens": 384,
        "gen": lambda: _sys_simple(
            "You are a research scientist. You think step by step. "
            "You consider multiple perspectives before answering. "
            "You cite sources when possible. You acknowledge uncertainty. "
            "You structure responses with reasoning chains."
        )() + [
            {"role": "user", "content": "Design a distributed key-value store with strong consistency."},
            {"role": "user", "content": "How would you handle network partitions?"},
            {"role": "user", "content": "Compare Raft vs Paxos for this design."},
            {"role": "user", "content": "What's the throughput bottleneck in your design?"},
            {"role": "user", "content": "How would you scale to 1000 nodes?"},
            {"role": "user", "content": "Analyze the failure modes of your system."},
        ],
    },
]


def main():
    print("=" * 65)
    print(f" SMART KV - Multi-Trial Data Collector ({N_TRIALS} trials)")
    print("=" * 65)

    # Check server
    try:
        data = _api_get("/health", timeout=5)
        print(f" Server: {BASE_URL} (OK)")
    except Exception as e:
        print(f" ERROR: Server not reachable: {e}")
        print(" Start with:  start.bat test")
        sys.exit(1)

    # Clear old exported files
    for f in glob.glob("training_data_*.bin"):
        os.remove(f)
        print(f" Removed old: {f}")

    # Get model info (for display only — use "default" as model name)
    try:
        data = _api_get("/v1/models", timeout=5)
        global MODEL_NAME
        model_display = data.get("data", [{}])[0].get("id", "unknown")
        MODEL_NAME = "default"
        print(f" Model: {model_display}")
    except:
        print(" Model: (unknown)")
        MODEL_NAME = "default"

    print(f" Scenarios: {len(SCENARIOS)}")
    print(f" Trials:    {N_TRIALS}")
    print()

    total_est_tokens = 0

    for trial in range(N_TRIALS):
        print(f"{'-' * 65}")
        print(f" Trial {trial + 1} / {N_TRIALS}")
        print(f"{'-' * 65}")

        # Warm-up: cold-start query to initialize GPU/cache state
        if trial == 0:
            print("  Warm-up...", end=" ", flush=True)
            _, pt = send_chat([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'ready'."},
            ], max_tokens=8)
            print(f"{pt} tokens")
            time.sleep(1)

        trial_tokens = 0

        for si, scenario in enumerate(SCENARIOS):
            name = scenario["name"]
            print(f"  [{si+1:2d}/{len(SCENARIOS)}] {name:20s}...", end=" ", flush=True)
            tokens = run_scenario(scenario)
            trial_tokens += tokens
            print(f"{tokens:5d} tokens")

        total_est_tokens += trial_tokens
        avg_per_turn = trial_tokens // sum(len(s["gen"]()) for s in SCENARIOS)
        print(f"  => Trial total: {trial_tokens:,} tokens (~{avg_per_turn} per turn)")

    # Show results
    print(f"\n{'=' * 65}")
    print(" COLLECTION COMPLETE")
    print(f"{'=' * 65}")
    print(f" Total tokens processed: {total_est_tokens:,}")

    data_files = sorted(glob.glob("training_data_*.bin"))
    if data_files:
        print()
        for f in data_files:
            size = os.path.getsize(f)
            n = max(0, (size - 8) // 68)
            print(f"  {f}: {size:>8,} bytes  (~{n:,} samples)")
        best = max(data_files, key=lambda f: os.path.getsize(f))
        print(f"\n Best dataset: {best}")
        n_best = max(0, (os.path.getsize(best) - 8) // 68)
        print(f" Samples:      ~{n_best:,}")
        print(f"\n Train:")
        print(f"  python train_learned_score.py {best}")
    else:
        print("\n No exported files found. The plugin auto-exports at 1000 and 4000 samples.")
        print(" You may need more trials or longer scenarios." if N_TRIALS < 10 else " Check that the plugin loaded correctly.")

    print()


if __name__ == "__main__":
    main()
