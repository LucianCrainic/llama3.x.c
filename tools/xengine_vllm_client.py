#!/usr/bin/env python3
"""Copy of the setup client (verified in the vLLM setup job), plus --ids-file: exact token ids from a file, so every
engine is timed on the same ids. Single-request streaming benchmark client for `vllm serve` (/v1/completions).

Prompts are token-id lists of an exact length built from a fixed text with the
model's tokenizer (BOS + text tokens, truncated or repeated to the exact count).
Greedy, ignore_eos, max_tokens = G.  One JSON object per request on stdout/--out.
Only stdlib + `tokenizers` (or `transformers`) are needed.

Workloads: P (2048 in, 1 out), D (128 in, 128 out), L128 (8192 in, 128 out),
L1 (8192 in, 1 out), or custom --in N --gen G.  Warm-ups are recorded with
"warmup": true and must be dropped by the analysis.
"""
import argparse, http.client, json, sys, time, urllib.parse, os

WORKLOADS = {"P": (2048, 1), "D": (128, 128), "L128": (8192, 128), "L1": (8192, 1)}
BASE_TEXT = (
    "The history of computing is a story of increasingly clever ways to move numbers "
    "between memory and arithmetic units. Early machines stored a few kilobytes in "
    "mercury delay lines, while modern processors stack gigabytes of high bandwidth "
    "memory next to the cores. A language model generates text one token at a time, "
    "and for each token the processor must read every weight of the network once, so "
    "the speed of generation is limited by how fast memory can be streamed. Engineers "
    "therefore quantize weights to eight bits, pin threads to cores, place pages on the "
    "memory attached to the socket that uses them, and measure everything twice. "
    "In the evening the harbour of Naples fills with light, and the ferries to the "
    "islands leave one after another across the quiet water. ")


def load_tokenizer(path):
    try:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        return lambda s: tk.encode(s, add_special_tokens=False).ids
    except Exception:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(path)
        return lambda s: tk.encode(s, add_special_tokens=False)


def make_ids(enc, n, bos, text_file=None, offset=0):
    text = open(text_file).read() if text_file else BASE_TEXT
    base = enc(text)
    assert base, "empty tokenization"
    if offset: base = base[offset % len(base):] + base[:offset % len(base)]  # different prompt, same length
    ids = [bos]
    while len(ids) < n:
        ids += base
    return ids[:n]


def one_request(host, port, model, ids, gen, timeout, text_prompt=None):
    body = {"model": model, "prompt": text_prompt if text_prompt is not None else ids,
            "max_tokens": gen, "min_tokens": gen, "temperature": 0, "stream": True,
            "ignore_eos": True, "return_token_ids": True,
            "stream_options": {"include_usage": True}}
    payload = json.dumps(body).encode()
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    t_send = time.perf_counter()
    conn.request("POST", "/v1/completions", payload, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    rec = {"t_send": 0.0, "http_status": resp.status}
    arrivals, text, tok_ids, usage = [], [], [], None
    first = None
    if resp.status != 200:
        rec["error"] = resp.read().decode(errors="replace")[:500]
        return rec
    buf = b""
    done = False
    while not done:
        line = resp.readline()
        if not line:
            break
        now = time.perf_counter()
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        obj = json.loads(data)
        if obj.get("usage"):
            usage = obj["usage"]
        for ch in obj.get("choices", []):
            t = ch.get("text", "")
            ti = ch.get("token_ids") or []
            if t or ti:
                n_new = max(len(ti), 1 if t else 0)
                arrivals.extend([now - t_send] * n_new)
                if first is None:
                    first = now - t_send
                text.append(t)
                tok_ids.extend(ti)
    t_end = time.perf_counter() - t_send
    conn.close()
    rec.update({
        "t_first_token": first, "token_arrivals": arrivals, "t_end": t_end,
        "n_chunk_tokens": len(arrivals),
        "usage_prompt_tokens": (usage or {}).get("prompt_tokens"),
        "usage_completion_tokens": (usage or {}).get("completion_tokens"),
        "text": "".join(text), "token_ids": tok_ids,
    })
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="llama31-8b-w8a8", help="served model name")
    ap.add_argument("--tokenizer", required=True, help="dir with tokenizer.json")
    ap.add_argument("--workload", default="D", help="P, D, L128, L1, comma list, or 'custom'")
    ap.add_argument("--in-tokens", type=int); ap.add_argument("--gen", type=int)
    ap.add_argument("--repeats", type=int, default=3, help="measured repeats")
    ap.add_argument("--warmups", type=int, default=1, help="discarded warm-up requests per workload")
    ap.add_argument("--bos", type=int, default=128000)
    ap.add_argument("--offset", type=int, default=0, help="rotate the base text by this many tokens (gives a different prompt of the same length)")
    ap.add_argument("--text-file", help="fixed text for building ids (default: built-in paragraph)")
    ap.add_argument("--ids-file", help="JSON array of prompt token ids (BOS first), used as is; length = prompt tokens")
    ap.add_argument("--text-prompt", help="send this plain-text prompt instead of ids (sanity check)")
    ap.add_argument("--out", help="JSONL output file (appended)")
    ap.add_argument("--tag", default="", help="free label stored in every record")
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args()

    out = open(a.out, "a") if a.out else None
    def emit(r):
        s = json.dumps(r)
        print(s, flush=True)
        if out: out.write(s + "\n"); out.flush()

    fixed_ids = json.load(open(a.ids_file)) if a.ids_file else None
    if fixed_ids is not None:
        wl = [(a.workload, len(fixed_ids), a.gen)]
    elif a.text_prompt is not None:
        wl = [("sanity", 0, a.gen or 32)]
    elif a.workload == "custom":
        wl = [("custom", a.in_tokens, a.gen)]
    else:
        wl = [(w, *WORKLOADS[w]) for w in a.workload.split(",")]
    enc = None if (a.text_prompt is not None or fixed_ids is not None) else load_tokenizer(a.tokenizer)
    for name, n_in, gen in wl:
        ids = fixed_ids if fixed_ids is not None else (
            None if a.text_prompt is not None else make_ids(enc, n_in, a.bos, a.text_file, a.offset))
        for i in range(a.warmups + a.repeats):
            r = one_request(a.host, a.port, a.model, ids, gen, a.timeout, a.text_prompt)
            r.update({"workload": name, "prompt_tokens_sent": n_in, "gen_requested": gen,
                      "repeat": i, "warmup": i < a.warmups, "tag": a.tag,
                      "wall_time": time.time()})
            if r.get("token_arrivals") and len(r["token_arrivals"]) > 1:
                arr = r["token_arrivals"]
                r["tpot_mean_s"] = (arr[-1] - arr[0]) / (len(arr) - 1)
                r["decode_tok_per_s"] = (len(arr) - 1) / (arr[-1] - arr[0]) if arr[-1] > arr[0] else None
            r["ok"] = (r.get("http_status") == 200
                       and (a.text_prompt is not None or r.get("usage_prompt_tokens") == n_in)
                       and r.get("usage_completion_tokens") == gen)
            emit(r)
            if "error" in r:
                print("ERROR", r["error"], file=sys.stderr); sys.exit(1)

if __name__ == "__main__":
    main()
