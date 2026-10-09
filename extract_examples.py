#!/usr/bin/env python3
"""
Extract the most interesting false-negative examples for the qualitative
analysis writeup (analysis/false_negative_examples.md).

Heuristic: prefer examples where the BM25 candidate looks topically very
similar to the positive (high lexical overlap with the query); those are
the cleanest "obvious false negative" cases for the report.

Usage
-----
    python extract_examples.py --n 20
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter

SCORES_PATH = Path("data/verifier_scores.jsonl")
TRAIN_PATH  = Path("data/train.json")
OUT_PATH    = Path("analysis/false_negative_examples.md")

def overlap_score(query: str, passage: str) -> float:
    qw = set(query.lower().split())
    pw = set(passage.lower().split())
    if not qw:
        return 0.0
    return len(qw & pw) / len(qw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    # Build query -> positive map for context
    with TRAIN_PATH.open() as f:
        positives = {ex["query"]: ex["positive"] for ex in json.load(f)}

    flagged = []
    with SCORES_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("is_false_negative"):
                rec["overlap"] = overlap_score(rec["query"], rec["candidate_text"])
                flagged.append(rec)

    flagged.sort(key=lambda r: r["overlap"], reverse=True)
    chosen = flagged[: args.n]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        f.write("# Qualitative False-Negative Examples\n\n")
        f.write(f"Top {len(chosen)} of {len(flagged)} flagged false negatives, "
                "ranked by query-passage word overlap.\n\n")
        f.write("Each example shows: the query, the *labeled* positive passage, "
                "and a passage that the LLM verifier judged as ALSO answering the "
                "query (i.e. a false negative that was being used to push the model "
                "in the wrong direction during contrastive training).\n\n")
        f.write("---\n\n")
        for i, r in enumerate(chosen, 1):
            pos = positives.get(r["query"], "(positive not found)")
            f.write(f"## Example {i}\n\n")
            f.write(f"**Query:** {r['query']}\n\n")
            f.write(f"**Labeled positive:**\n\n> {pos[:500]}{'...' if len(pos) > 500 else ''}\n\n")
            f.write(f"**False negative (flagged by LLM):**\n\n> "
                    f"{r['candidate_text'][:500]}"
                    f"{'...' if len(r['candidate_text']) > 500 else ''}\n\n")
            f.write(f"_LLM raw response:_ `{r.get('raw_response', '')[:60]}`  ")
            f.write(f"_word overlap:_ {r['overlap']:.2f}\n\n")
            f.write("---\n\n")
    print(f"[write] {OUT_PATH} ({len(chosen)} examples)")

if __name__ == "__main__":
    main()
