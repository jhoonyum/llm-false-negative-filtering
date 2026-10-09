#!/usr/bin/env python3
"""
False-Negative Verifier v2 (Person 3)
=====================================
v2 fixes the empty-response bug caused by Qwen3.6's thinking mode.

Changes vs v1:
  - Adds `think=False` to the Ollama chat call (Qwen3 thinking off)
  - Adds `/no_think` directive to system prompt (belt-and-suspenders)
  - Raises num_predict from 4 to 16 (small buffer)
  - parse_answer() now strips <think>...</think> blocks if they leak through

Usage same as v1:
    python run_verifier.py --smoke
    python run_verifier.py --workers 4
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import ollama
from tqdm import tqdm

DEFAULT_INPUT  = "data/train_data_for_LLM.json"
DEFAULT_OUTPUT = "data/verifier_scores.jsonl"
DEFAULT_MODEL  = "qwen3.6:35b"

# /no_think is Qwen3's directive to disable chain-of-thought output
SYSTEM_PROMPT = (
    "/no_think You are an expert relevance judge for information retrieval. "
    "Given a query and a passage, decide whether the passage answers or "
    "contains the information sought by the query. "
    "Reply with exactly one word: Yes or No. No explanation, no thinking."
)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def build_user_prompt(query: str, passage: str) -> str:
    if len(passage) > 1500:
        passage = passage[:1500] + " [...]"
    return (
        f"Query: {query}\n\n"
        f"Passage: {passage}\n\n"
        "Does this passage answer the query? "
        "Answer with one word only: Yes or No. /no_think"
    )

def parse_answer(text: str) -> str:
    """Map raw model output to 'Yes' / 'No' / 'Unknown'.
    Robust to leftover <think>...</think> blocks and various formattings."""
    if not text:
        return "Unknown"
    # Strip any thinking block that leaked through
    cleaned = THINK_RE.sub("", text).strip()
    # Drop common prefix decoration
    cleaned = cleaned.lstrip("'\"`*_ \t\n.,:;-")
    low = cleaned.lower()
    if low.startswith("yes"):
        return "Yes"
    if low.startswith("no"):
        return "No"
    head = low[:60]
    has_yes = "yes" in head
    has_no  = "no"  in head
    if has_yes and not has_no:
        return "Yes"
    if has_no and not has_yes:
        return "No"
    return "Unknown"

def call_ollama(client: ollama.Client, model: str, query: str, passage: str):
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(query, passage)},
        ],
        think=False,   # Ollama 0.22+: disables Qwen3 thinking mode
        options={
            "temperature": 0.0,
            "num_predict": 16,   # small buffer in case thinking still leaks
            "top_p": 1.0,
            "seed": 42,
        },
    )
    raw = response["message"]["content"]
    return raw, parse_answer(raw)

def load_done_keys(output_path: Path) -> set[tuple[int, int]]:
    done: set[tuple[int, int]] = set()
    if not output_path.exists():
        return done
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Only count records with a real answer; let Unknown rows be redone
                if obj.get("llm_answer") in ("Yes", "No"):
                    done.add((obj["query_idx"], obj["candidate_idx"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",   default=DEFAULT_INPUT)
    p.add_argument("--output",  default=DEFAULT_OUTPUT)
    p.add_argument("--model",   default=DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--smoke",   action="store_true")
    p.add_argument("--limit",   type=int, default=None)
    args = p.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {in_path}")
    with in_path.open() as f:
        data = json.load(f)
    print(f"[load] {len(data)} examples")

    if args.smoke:
        data = data[:10]
        print(f"[smoke] using first {len(data)} examples")
    elif args.limit:
        data = data[: args.limit]

    tasks: list[tuple[int, int, str, str]] = []
    for q_idx, ex in enumerate(data):
        for c_idx, cand in enumerate(ex["bm25_candidates"]):
            tasks.append((q_idx, c_idx, ex["query"], cand))
    total_initial = len(tasks)

    done = load_done_keys(out_path)
    if done:
        print(f"[resume] {len(done)} pairs already done (Yes/No only); skipping")
    tasks = [t for t in tasks if (t[0], t[1]) not in done]
    print(f"[plan] {len(tasks)} pairs to process (of {total_initial})")
    if not tasks:
        print("[done] nothing to do.")
        return

    client = ollama.Client()

    print(f"[warmup] loading {args.model}...")
    t0 = time.time()
    try:
        warm = client.chat(
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": "Query: test\n\nPassage: test\n\nDoes this passage answer the query? Answer Yes or No. /no_think"},
            ],
            think=False,
            options={"num_predict": 8, "temperature": 0.0},
        )
        print(f"[warmup] raw response: {warm['message']['content']!r}")
    except TypeError as e:
        # Older ollama-python doesn't support `think=` kwarg
        print(f"[warmup] note: 'think' kwarg not supported by your ollama-python; "
              f"relying on /no_think directive. ({e})")
        # Fall back: monkey-patch call_ollama to drop think kwarg
        def _call_no_think(client, model, query, passage):
            response = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(query, passage)},
                ],
                options={"temperature": 0.0, "num_predict": 16, "top_p": 1.0, "seed": 42},
            )
            raw = response["message"]["content"]
            return raw, parse_answer(raw)
        global call_ollama
        call_ollama = _call_no_think
    except Exception as e:
        print(f"[error] warmup failed: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"[warmup] done in {time.time() - t0:.1f}s")

    def process(task):
        q_idx, c_idx, query, candidate = task
        try:
            raw, label = call_ollama(client, args.model, query, candidate)
            return {
                "query_idx":     q_idx,
                "candidate_idx": c_idx,
                "query":         query,
                "candidate_text": candidate,
                "raw_response":  raw,
                "llm_answer":    label,
                "is_false_negative": label == "Yes",
            }
        except Exception as e:
            return {
                "query_idx":     q_idx,
                "candidate_idx": c_idx,
                "query":         query,
                "candidate_text": candidate,
                "raw_response":  None,
                "llm_answer":    "Unknown",
                "is_false_negative": False,
                "error":         str(e),
            }

    yes_n = no_n = unk_n = err_n = 0
    pbar = tqdm(total=len(tasks), desc="verify", smoothing=0.05)

    with out_path.open("a") as fout:
        if args.workers <= 1:
            for t in tasks:
                rec = process(t)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                lbl = rec.get("llm_answer")
                yes_n += (lbl == "Yes")
                no_n  += (lbl == "No")
                unk_n += (lbl == "Unknown")
                err_n += int("error" in rec)
                pbar.update(1)
                pbar.set_postfix(yes=yes_n, no=no_n, unk=unk_n, err=err_n)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(process, t) for t in tasks]
                for fut in as_completed(futs):
                    rec = fut.result()
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    lbl = rec.get("llm_answer")
                    yes_n += (lbl == "Yes")
                    no_n  += (lbl == "No")
                    unk_n += (lbl == "Unknown")
                    err_n += int("error" in rec)
                    pbar.update(1)
                    pbar.set_postfix(yes=yes_n, no=no_n, unk=unk_n, err=err_n)

    pbar.close()
    total = yes_n + no_n + unk_n
    print(f"\n[done] Yes={yes_n} ({yes_n/total*100:.1f}%)  "
          f"No={no_n} ({no_n/total*100:.1f}%)  "
          f"Unknown={unk_n} ({unk_n/total*100:.1f}%)  "
          f"errors={err_n}")
    print(f"[done] output: {out_path}")
    if unk_n > total * 0.05:
        print("[warn] >5% Unknown: inspect raw_response field, prompt may need tuning.")

if __name__ == "__main__":
    main()
