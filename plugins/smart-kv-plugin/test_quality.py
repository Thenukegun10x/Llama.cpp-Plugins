"""PPL comparison: generate F16 reference, then measure continuation PPL on identical context."""
import json, urllib.request, math, os

URL = "http://127.0.0.1:12345"
PROMPT = "Write a complete distributed KV store in Python with consistent hashing, Raft consensus, and WAL logging. 800+ lines of real code."
CONTINUE = "Now add unit tests for the consensus module."

def chat(messages, max_tok=1):
    data = json.dumps({
        "messages": messages, "max_tokens": max_tok,
        "temperature": 0.0, "seed": 42,
        "logprobs": True, "top_logprobs": 1,
    }).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data,
        {"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())

def main():
    ref_file = "f16_ref.txt"

    if not os.path.exists(ref_file):
        print("[1/2] Generating F16 reference...", flush=True)
        resp = chat([{"role": "user", "content": PROMPT}], max_tok=1024)
        text = resp["choices"][0]["message"]["content"]
        with open(ref_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"       Saved {ref_file} ({len(text)} chars)")
        ppl, n = 0, 0
        for tok in resp["choices"][0].get("logprobs", {}).get("content", []):
            ppl += -tok["logprob"]; n += 1
        print(f"       Generation PPL={math.exp(ppl/n):.4f} ({n} tokens)")
    else:
        with open(ref_file, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"[1/2] Using cached {ref_file} ({len(text)} chars)")

    # Measure continuation PPL — context is the F16 reference text,
    # continuation is deterministic (temp=0, seed=42) across profiles
    print("[2/2] Measuring continuation PPL...", end=" ", flush=True)
    resp = chat([
        {"role": "user", "content": PROMPT},
        {"role": "assistant", "content": text},
        {"role": "user", "content": CONTINUE},
    ], max_tok=128)

    lp = resp["choices"][0].get("logprobs", {}).get("content", [])
    if lp:
        nll = sum(-t.get("logprob", 0) for t in lp)
        ppl = math.exp(nll / len(lp))
        print(f"PPL={ppl:.4f} ({len(lp)} tokens)")
    else:
        print("no logprobs")

if __name__ == "__main__":
    main()
