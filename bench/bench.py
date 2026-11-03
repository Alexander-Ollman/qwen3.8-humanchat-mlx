#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible chat endpoint. Report latency and throughput.

Client measurements use streaming time stamps. Server measurements use the
llama-server `timings` object when the server sends it. Ollama does not send
it, so those fields stay empty there. Standard library only.
"""
import argparse
import json
import statistics as st
import threading
import time
import urllib.request

FILLER = (
    "The harbor office logs each ship by name, tonnage, cargo class, and arrival "
    "tide. Clerks copy the entry to the district ledger before the next watch. "
)


GLOBAL_EXTRA = {}


def post_stream(base, model, messages, max_tokens, extra=None):
    """Send one streaming request. Return one result dict."""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "timings_per_token": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body.update(GLOBAL_EXTRA)
    body.update(extra or {})
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    stamps = []
    usage = timings = None
    finish = None
    text = []
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            usage = obj.get("usage") or usage
            timings = obj.get("timings") or timings
            for ch in obj.get("choices") or []:
                finish = ch.get("finish_reason") or finish
                delta = ch.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content")
                if piece:
                    stamps.append(time.perf_counter())
                    text.append(piece)
    t_end = time.perf_counter()
    if not stamps:
        raise RuntimeError(f"no tokens in response, finish={finish}")
    ttft = stamps[0] - t0
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    out_tokens = (usage or {}).get("completion_tokens") or len(stamps)
    decode_s = t_end - stamps[0]
    return {
        "ttft_s": ttft,
        "total_s": t_end - t0,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": out_tokens,
        "client_decode_tps": (out_tokens - 1) / decode_s if decode_s > 0 else None,
        "itl_p50_ms": st.median(gaps) * 1000 if gaps else None,
        "itl_p99_ms": pct(gaps, 99) * 1000 if gaps else None,
        "server_prompt_n": (timings or {}).get("prompt_n"),
        "server_cache_n": (timings or {}).get("cache_n"),
        "server_prompt_tps": (timings or {}).get("prompt_per_second"),
        "server_decode_tps": (timings or {}).get("predicted_per_second"),
        "finish": finish,
        "text_head": "".join(text)[:120],
    }


def pct(values, p):
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))
    return ordered[idx]


def nonce_prompt(target_chars, tag):
    """Build a prompt with a unique head, so the prompt cache cannot match it."""
    head = f"Session key {tag}-{time.time_ns()}. "
    body = (FILLER * (target_chars // len(FILLER) + 1))[:target_chars]
    return head + body + "\nWrite a long, detailed report about harbor logistics."


def user(text):
    return [{"role": "user", "content": text}]


def summarize(rows, keys):
    out = {"n": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            out[k] = {
                "median": round(st.median(vals), 3),
                "min": round(min(vals), 3),
                "max": round(max(vals), 3),
            }
    return out


KEYS = [
    "ttft_s", "total_s", "prompt_tokens", "completion_tokens",
    "client_decode_tps", "itl_p50_ms", "itl_p99_ms",
    "server_prompt_tps", "server_decode_tps",
]


def run_cold_latency(a):
    """Measure TTFT with an uncached prompt at each prompt size."""
    res = {}
    for chars in a.prompt_chars:
        rows = []
        for i in range(a.reps):
            rows.append(post_stream(a.base, a.model, user(nonce_prompt(chars, "cold")), 32))
        res[str(chars)] = {"summary": summarize(rows, KEYS), "rows": rows}
        print("cold", chars, json.dumps(res[str(chars)]["summary"]), flush=True)
    return res


def run_warm_latency(a):
    """Measure TTFT when a long shared prefix is already in the prompt cache."""
    base_text = nonce_prompt(a.warm_chars, "warm")
    seed = post_stream(a.base, a.model, user(base_text + "\nQuestion 0: say ok."), 8)
    rows = []
    for i in range(1, a.reps + 1):
        msg = user(base_text + f"\nQuestion {i}: name one color, then the number {i * 37}.")
        rows.append(post_stream(a.base, a.model, msg, 32))
    res = {"seed": seed, "summary": summarize(rows, KEYS + ["server_cache_n"]), "rows": rows}
    print("warm", json.dumps(res["summary"]), flush=True)
    return res


def run_decode(a):
    """Measure sustained decode speed at a fixed output length."""
    rows = []
    for i in range(a.reps):
        msg = user(nonce_prompt(400, "decode"))
        rows.append(post_stream(a.base, a.model, msg, a.decode_tokens, {"ignore_eos": True}))
    res = {"summary": summarize(rows, KEYS), "rows": rows}
    print("decode", json.dumps(res["summary"]), flush=True)
    return res


def run_concurrency(a):
    """Measure aggregate output tokens per second at each concurrency level."""
    res = {}
    for level in a.concurrency:
        rows, errors = [], []
        lock = threading.Lock()

        def worker(idx):
            try:
                msg = user(nonce_prompt(800, f"c{level}-{idx}"))
                r = post_stream(a.base, a.model, msg, a.conc_tokens, {"ignore_eos": True})
                with lock:
                    rows.append(r)
            except Exception as exc:
                with lock:
                    errors.append(repr(exc))

        t0 = time.perf_counter()
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(level)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        total = sum(r["completion_tokens"] for r in rows)
        res[str(level)] = {
            "wall_s": round(wall, 2),
            "total_tokens": total,
            "aggregate_tps": round(total / wall, 2),
            "requests_per_min": round(len(rows) / wall * 60, 2),
            "errors": errors,
            "summary": summarize(rows, KEYS),
        }
        s = res[str(level)]
        print("conc", level, json.dumps({k: s[k] for k in s if k != "summary"}),
              json.dumps(s["summary"].get("ttft_s")), json.dumps(s["summary"].get("client_decode_tps")),
              flush=True)
    return res


def run_smoke(a):
    """Check plain chat and tool calling through the OpenAI wire format."""
    def call(body):
        req = urllib.request.Request(
            a.base + "/chat/completions",
            data=json.dumps({"model": a.model, "max_tokens": 200, "temperature": 0,
                             "chat_template_kwargs": {"enable_thinking": False}, **body}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)["choices"][0]

    tool = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}]
    out = {
        "chat": call({"messages": user("hey, what's up?")}),
        "tool": call({"messages": user("what's the weather in Lisbon right now?"), "tools": tool}),
    }
    print("smoke", json.dumps(out)[:800], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--model", default="humanlike-2.0-nvfp4")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--prompt-chars", type=int, nargs="+", default=[200, 8000, 32000, 80000])
    ap.add_argument("--warm-chars", type=int, default=32000)
    ap.add_argument("--decode-tokens", type=int, default=512)
    ap.add_argument("--conc-tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--stages", nargs="+", default=["smoke", "cold", "warm", "decode", "conc"])
    ap.add_argument("--extra", default='{"reasoning_effort": "none"}', help="JSON fields to merge into each streaming request.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    GLOBAL_EXTRA.update(json.loads(a.extra))

    stages = {"smoke": run_smoke, "cold": run_cold_latency, "warm": run_warm_latency,
              "decode": run_decode, "conc": run_concurrency}
    out = {"args": vars(a), "started": time.strftime("%FT%T%z")}
    for name in a.stages:
        out[name] = stages[name](a)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
