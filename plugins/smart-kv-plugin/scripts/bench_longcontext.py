"""
Long-context cache quality benchmark for an already-running llama-server.

Distributes needle facts at every 10% of cache position, then queries them
to measure recall. Pass rate by position reveals where eviction hurts most.

Also records GPU VRAM usage at each query via Windows performance counters.

Run the server separately with either:
  start.bat f16
  start.bat smart-ultra

Then run this script once per profile and compare position-wise recall.
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request


BASE_URL = "http://127.0.0.1:12345"
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(PLUGIN_DIR, "..", ".."))


SYSTEM_PROMPT = "You are a helpful assistant."

FILLER_TOPICS = [
    ("geography", [
        "The capital of France is Paris, a major European cultural and economic center located on the Seine River.",
        "Mount Everest, at 8,848 meters, is the tallest mountain on Earth, located in the Himalayas between Nepal and Tibet.",
        "The Amazon River flows through Brazil, Peru, Colombia, and several other South American countries, draining an area of over 7 million square kilometers.",
        "Tokyo, Japan's capital, is the world's most populous metropolitan area with over 37 million residents.",
        "Australia is both a continent and a country, bordered by the Indian and Pacific Oceans, with Canberra as its capital.",
        "The Sahara Desert in North Africa is the largest hot desert in the world, covering approximately 9.2 million square kilometers.",
        "Antarctica is the Earth's southernmost continent, containing about 90% of the world's ice and 70% of its fresh water.",
        "Lake Baikal in Siberia, Russia, is the deepest lake in the world at 1,642 meters, containing about 20% of the world's unfrozen surface freshwater.",
        "The Great Barrier Reef off the coast of Queensland, Australia, is the largest coral reef system in the world, stretching over 2,300 kilometers.",
        "Istanbul is a transcontinental city straddling Europe and Asia across the Bosphorus Strait, making it strategically important throughout history.",
    ]),
    ("technology", [
        "The TCP/IP protocol suite, developed in the 1970s by Vint Cerf and Bob Kahn, forms the foundation of modern internet communication.",
        "Quantum computing leverages superposition and entanglement to perform certain calculations exponentially faster than classical computers.",
        "The Linux kernel, created by Linus Torvalds in 1991, now powers the majority of servers, supercomputers, and Android devices worldwide.",
        "Transformer architecture, introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017, revolutionized natural language processing.",
        "The C programming language, developed by Dennis Ritchie at Bell Labs, remains one of the most influential languages in systems programming.",
        "Microservices architecture decomposes applications into independently deployable services, each running its own process and communicating via lightweight mechanisms.",
        "Database indexing using B-trees provides O(log n) lookup time, making large-scale data retrieval feasible for modern applications.",
        "The concept of virtual memory allows a computer to use more memory than physically available by swapping data between RAM and disk storage.",
        "Rust is a systems programming language focused on safety and performance, eliminating data races through its ownership model.",
        "GraphQL, developed by Facebook, provides a flexible query language for APIs that lets clients request exactly the data they need.",
    ]),
    ("history", [
        "The Roman Empire, at its peak under Emperor Trajan in 117 CE, spanned from Britain to Mesopotamia, covering approximately 5 million square kilometers.",
        "The Industrial Revolution began in Great Britain around 1760, transforming manufacturing through mechanization, factory systems, and steam power.",
        "The Apollo 11 mission landed the first humans on the Moon on July 20, 1969, with Neil Armstrong and Buzz Aldrin spending about two hours on the lunar surface.",
        "The Magna Carta, signed in 1215, established the principle that everyone, including the monarch, was subject to the law.",
        "Leonardo da Vinci, active during the Italian Renaissance, was a polymath whose contributions spanned painting, engineering, anatomy, and architecture.",
        "The Berlin Wall fell on November 9, 1989, marking a pivotal moment in the dissolution of the Soviet Union and the end of the Cold War.",
        "The Black Death pandemic in the 14th century killed an estimated 75-200 million people across Europe, Asia, and North Africa.",
        "The Rosetta Stone, discovered in 1799, was key to deciphering Egyptian hieroglyphs as it contained the same text in Greek, Demotic, and hieroglyphic scripts.",
        "The Spanish conquest of the Aztec Empire, led by Hernan Cortes between 1519 and 1521, fundamentally altered the course of Mesoamerican history.",
        "The Library of Alexandria, one of the largest and most significant libraries of the ancient world, was likely destroyed gradually over several centuries.",
    ]),
    ("science", [
        "Photosynthesis is the process by which plants convert light energy into chemical energy, producing glucose and oxygen from carbon dioxide and water.",
        "The theory of general relativity, published by Albert Einstein in 1915, describes gravity as the curvature of spacetime caused by mass and energy.",
        "DNA, or deoxyribonucleic acid, carries the genetic instructions for the development and functioning of all known living organisms.",
        "The periodic table of elements, first organized by Dmitri Mendeleev in 1869, arranges chemical elements by atomic number and electron configuration.",
        "Natural selection, proposed by Charles Darwin, is the differential survival and reproduction of individuals due to differences in phenotype.",
        "The human brain contains approximately 86 billion neurons, each connected to thousands of others, forming trillions of synapses.",
        "Plate tectonics explains the movement of Earth's lithosphere, which is divided into several large and small plates that float on the asthenosphere.",
        "The electromagnetic spectrum spans from gamma rays with wavelengths less than 10 picometers to radio waves with wavelengths of kilometers.",
        "Entropy, a concept from thermodynamics, measures the disorder or randomness in a system and always increases in isolated systems.",
        "Vaccines work by exposing the immune system to a harmless form of a pathogen, enabling the body to develop immunity without causing disease.",
    ]),
    ("mathematics", [
        "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides.",
        "The Fibonacci sequence, where each number is the sum of the two preceding ones, appears frequently in nature, from pine cones to seashells.",
        "Euler's formula, e^(iπ) + 1 = 0, is considered one of the most beautiful equations in mathematics, connecting five fundamental constants.",
        "The P versus NP problem asks whether every problem whose solution can be quickly verified can also be quickly solved.",
        "The Riemann zeta function, central to the Riemann hypothesis, is defined as the sum of 1/n^s for complex numbers s with real part greater than 1.",
        "Bayes' theorem describes the probability of an event based on prior knowledge of conditions that might be related to the event.",
        "The Gaussian distribution, also called the normal distribution, is a probability distribution that is symmetric about the mean.",
        "Group theory studies algebraic structures known as groups, which abstract the properties of symmetry and are fundamental to modern algebra.",
        "The concept of limits in calculus formalizes the behavior of functions as inputs approach certain values, forming the basis of derivatives and integrals.",
        "Graph theory, originating from Euler's solution to the Seven Bridges of Königsberg problem, studies networks of vertices connected by edges.",
    ]),
]


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


def get_server_vram_mib():
    """Query llama-server process dedicated GPU memory in MiB via performance counters."""
    try:
        # Get server PID
        pid_result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-Process llama-server -ErrorAction SilentlyContinue | "
             "Select-Object -First 1 -ExpandProperty Id)"],
            capture_output=True, text=True, timeout=5
        )
        pid = pid_result.stdout.strip()
        if not pid:
            return 0.0

        # Get all GPU process memory counters
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' -ErrorAction SilentlyContinue | "
             "Select-Object -ExpandProperty CounterSamples | "
             "Format-Table InstanceName, CookedValue -HideTableHeaders"],
            capture_output=True, text=True, timeout=10
        )
        total = 0.0
        for line in result.stdout.strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                instance = parts[0]
                val_str = parts[-1]
                if pid in instance and val_str.isdigit():
                    total += float(val_str)
        return total / 1024 / 1024
    except Exception:
        return 0.0


def estimate_tokens(messages):
    chars = sum(len(m.get("content", "")) for m in messages)
    return max(1, int(chars / 3.0))


def chat_completion(base_url, messages, max_tokens, temperature, seed, timeout):
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    return api_post(base_url, "/v1/chat/completions", body, timeout=timeout)


def doc_text(doc_id, topic="mixed"):
    if topic == "mixed":
        topic, facts = random.choice(FILLER_TOPICS)
    else:
        facts = dict(FILLER_TOPICS).get(topic, FILLER_TOPICS[0][1])
    fact = facts[doc_id % len(facts)]
    return f"[Document {doc_id}] Topic: {topic}\n{fact}"


def needle_text(nid):
    code = f"CODE{nid:04d}{random.randint(1000, 9999)}"
    prefix = f"Remember: the code for query {nid} is {code}."
    return code, prefix


def run_benchmark(args):
    require_server(args.url)
    random.seed(args.seed)

    print(f"Long-context cache benchmark")
    print(f"  profile:          {args.profile}")
    print(f"  server:           {args.url}")
    print(f"  needles:          {args.needles}")
    print(f"  context fill:     {args.context_fill * 100:.0f}%")
    print(f"  ctx:              {args.ctx}")
    print()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_queries = 0
    recalled = 0
    records = []
    position_buckets = {}
    started = time.time()
    max_tokens = int(args.ctx * args.context_fill)

    def bucket(pos_pct):
        b = int(pos_pct * 10) * 10
        if b not in position_buckets:
            position_buckets[b] = {"found": 0, "total": 0}
        return b

    out_fp = open(args.out, "w", encoding="utf-8") if args.out else None

    initial_vram = get_server_vram_mib()
    print(f"  initial VRAM:   {initial_vram:.0f} MiB")

    def save_record(record):
        record["vram_mib"] = get_server_vram_mib()
        records.append(record)
        if out_fp:
            out_fp.write(json.dumps(record) + "\n")
            out_fp.flush()

    # Build the conversation: interleave fill docs + needles across the range
    doc_id = 0
    needles = []
    target_positions = [int(max_tokens * p) for p in
                        [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]]
    target_positions = target_positions[:args.needles]

    try:
        for ni, insert_at in enumerate(target_positions):
            # Fill until we reach the target position for this needle
            while estimate_tokens(messages) < insert_at and estimate_tokens(messages) < max_tokens:
                doc = doc_text(doc_id, args.topic)
                if estimate_tokens(messages) + len(doc) // 3 > max_tokens:
                    break
                messages.append({"role": "user", "content": doc})
                messages.append({
                    "role": "assistant",
                    "content": f"Understood. Processed document {doc_id}."
                })
                doc_id += 1

            if estimate_tokens(messages) >= max_tokens:
                break

            # Insert needle at this position
            code, ntext = needle_text(ni)
            pos_tokens = estimate_tokens(messages)
            pos_pct = pos_tokens / max(1, args.ctx)

            messages.append({"role": "user", "content": ntext})
            messages.append({
                "role": "assistant",
                "content": f"Stored. The code for query {ni} is saved."
            })

            needles.append({
                "id": ni,
                "code": code,
                "insertion_tokens": pos_tokens,
                "insertion_pct": pos_pct,
            })

        print(f"Built conversation: {doc_id} docs + {len(needles)} needles "
              f"({estimate_tokens(messages)} est tokens)")

        # Query each needle
        for n in needles:
            total_queries += 1

            query_msg = list(messages)
            query_msg.append({
                "role": "user",
                "content": f"What is the code for query {n['id']}? "
                           f"Reply with the exact code only, no explanations, no thinking."
            })

            try:
                body = {
                    "messages": query_msg,
                    "max_tokens": 2048,
                    "temperature": args.temperature,
                    "seed": args.seed + total_queries,
                    "max_thinking_tokens": 0 if args.suppress_reasoning else None,
                }
                if body["max_thinking_tokens"] is None:
                    del body["max_thinking_tokens"]
                response = api_post(args.url, "/v1/chat/completions", body, timeout=args.request_timeout)
                choice = response["choices"][0]
                msg = choice.get("message", {})
                raw = (msg.get("content", "") or "").strip()
                # Reasoning models put the answer in reasoning_content, extract from there too
                rc = (msg.get("reasoning_content", "") or "")
                finish = choice.get("finish_reason", "")
                usage = response.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                # Combine content + reasoning_content for code matching
                combined = raw + "\n" + rc
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                print(f"  ERROR [{n['id']}] at {n['insertion_pct'] * 100:.0f}%: HTTP {e.code}: {body[:120]}")
                save_record({
                    "profile": args.profile,
                    "needle": n["id"],
                    "code": n["code"],
                    "insertion_pct": round(n["insertion_pct"], 3),
                    "position_bucket": bucket(n["insertion_pct"]),
                    "recalled": False,
                    "response": f"HTTP_ERROR: {body[:200]}",
                    "prompt_tokens": 0,
                })
                continue
            except Exception as e:
                print(f"  ERROR [{n['id']}] at {n['insertion_pct'] * 100:.0f}%: {e}")
                save_record({
                    "profile": args.profile,
                    "needle": n["id"],
                    "code": n["code"],
                    "insertion_pct": round(n["insertion_pct"], 3),
                    "position_bucket": bucket(n["insertion_pct"]),
                    "recalled": False,
                    "response": f"EXCEPTION: {e}",
                    "prompt_tokens": 0,
                })
                continue

            ok = n["code"] in combined
            b = bucket(n["insertion_pct"])
            position_buckets[b]["total"] += 1
            if ok:
                recalled += 1
                position_buckets[b]["found"] += 1

            label = "RECALL" if ok else "MISS"
            preview = (raw or rc)[:80].replace("\n", " ")
            print(f"  {label} [code {n['id']}] at ~{n['insertion_pct'] * 100:.0f}% cache "
                  f"({position_buckets[b]['found']}/{position_buckets[b]['total']}) "
                  f"resp=\"{preview}\" "
                  f"({prompt_tokens} tokens)")

            save_record({
                "profile": args.profile,
                "needle": n["id"],
                "code": n["code"],
                "insertion_pct": round(n["insertion_pct"], 3),
                "position_bucket": b,
                "recalled": ok,
                "response": (raw or rc)[:200],
                "reasoning": rc[:500],
                "prompt_tokens": prompt_tokens,
            })

    finally:
        if out_fp:
            out_fp.close()

    elapsed = time.time() - started
    final_vram = get_server_vram_mib()
    print()
    print("Result")
    print(f"  profile:    {args.profile}")
    print(f"  recall:     {recalled}/{total_queries} = {100.0 * recalled / max(1, total_queries):.1f}%")
    print(f"  elapsed:    {elapsed:.1f}s")
    print(f"  VRAM:       {initial_vram:.0f} MiB -> {final_vram:.0f} MiB (delta {final_vram - initial_vram:+.0f} MiB)")
    print()
    print("  Recall by cache position:")
    print(f"  {'Position':<18} {'Found':>5} / {'Total':<5} {'Rate':>7}")
    print(f"  {'─' * 36}")
    for pct in sorted(position_buckets.keys()):
        b = position_buckets[pct]
        rate = 100.0 * b["found"] / max(1, b["total"])
        bar = "█" * int(rate / 5) + "░" * (20 - int(rate / 5))
        print(f"  {f'{pct}-{pct + 10}%':<18} {b['found']:>5} / {b['total']:<5} {rate:>6.1f}% {bar}")

    if args.out:
        print(f"\n  wrote: {args.out}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Long-context cache quality benchmark for a running llama-server."
    )
    parser.add_argument("--url", default=BASE_URL, help="llama-server base URL")
    parser.add_argument("--profile", default="manual", help="label stored in output, e.g. f16 or smart-ultra")
    parser.add_argument("--ctx", type=int, default=32768, help="context window size")
    parser.add_argument("--needles", type=int, default=9, help="needle facts (max 9)")
    parser.add_argument("--context-fill", type=float, default=0.85, help="target cache fill ratio")
    parser.add_argument("--topic", default="mixed", help="filler topic")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--out", help="write per-query JSONL results")
    parser.add_argument("--suppress-reasoning", action="store_true", default=True,
                        help="try to suppress model reasoning/thinking tokens")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run_benchmark(parse_args())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
