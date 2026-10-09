#!/usr/bin/env python3
"""
Summarize the verifier output (added when this repo was prepared)
=================================================================
Prints the counts quoted in the README from data/verifier_scores.jsonl.

    --train data/train.json   also replays build_filtered_train.py and reports
                              how many flagged passages were in that query's
                              own negatives list (the only ones it can remove)
    --figure PATH             draws the flagged-candidates-per-query chart

Usage
-----
    python analysis/summarize_scores.py
    python analysis/summarize_scores.py --train data/train.json --figure figures/flags_per_query.png
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_SCORES = "data/verifier_scores.jsonl"


def load_scores(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def overlap_score(query: str, passage: str) -> float:
    """Same measure as extract_examples.py: share of query words found in the passage."""
    qw = set(query.lower().split())
    pw = set(passage.lower().split())
    return len(qw & pw) / len(qw) if qw else 0.0


def pct(a: int, b: int) -> str:
    return f"{a / b * 100:.1f}%" if b else "n/a"


def summarize(rows: list[dict]) -> Counter:
    n = len(rows)
    answers = Counter(r["llm_answer"] for r in rows)
    print(f"pairs judged: {n:,}")
    for k in ("Yes", "No", "Unknown"):
        print(f"  {k:<7} {answers.get(k, 0):>6,}  {pct(answers.get(k, 0), n)}")
    print(f"  rows with an error field: {sum('error' in r for r in rows)}")

    n_cand = Counter(r["query_idx"] for r in rows)
    n_yes = Counter()
    for r in rows:
        n_yes[r["query_idx"]] += r["llm_answer"] == "Yes"
    queries = sorted(n_cand)
    q = len(queries)
    with_yes = sum(1 for k in queries if n_yes[k] > 0)
    all_yes = sum(1 for k in queries if n_yes[k] == n_cand[k])
    print(f"queries covered: {q:,}")
    print(f"  candidates per query: {dict(sorted(Counter(n_cand.values()).items()))}")
    print(f"  with at least one Yes: {with_yes:,} ({pct(with_yes, q)})")
    print(f"  with every candidate Yes: {all_yes:,} ({pct(all_yes, q)})")
    dist = Counter(n_yes[k] for k in queries)
    print("Yes answers per query:")
    for k in range(max(dist) + 1):
        print(f"  {k}: {dist.get(k, 0):>5,}  {pct(dist.get(k, 0), q)}")

    print("Yes rate by position in the BM25 candidate list:")
    by_pos = defaultdict(lambda: [0, 0])
    for r in rows:
        by_pos[r["candidate_idx"]][0] += 1
        by_pos[r["candidate_idx"]][1] += r["llm_answer"] == "Yes"
    for k in sorted(by_pos):
        tot, yes = by_pos[k]
        print(f"  {k + 1}: {yes:>5,} / {tot:>5,}  {pct(yes, tot)}")

    print("Yes rate by share of query words found in the passage:")
    bins = [(0.0, 0.25, "0-24%"), (0.25, 0.5, "25-49%"), (0.5, 0.75, "50-74%"),
            (0.75, 1.0, "75-99%"), (1.0, 1.01, "100%")]
    by_bin = defaultdict(lambda: [0, 0])
    for r in rows:
        o = overlap_score(r["query"], r["candidate_text"])
        for lo, hi, label in bins:
            if lo <= o < hi:
                by_bin[label][0] += 1
                by_bin[label][1] += r["llm_answer"] == "Yes"
                break
    for _, _, label in bins:
        tot, yes = by_bin[label]
        print(f"  {label:>6}: {yes:>5,} / {tot:>6,}  {pct(yes, tot)}")
    return dist


def replay_filter(rows: list[dict], train_path: Path) -> None:
    """Mirror build_filtered_train.py without writing anything."""
    with train_path.open(encoding="utf-8") as f:
        train = json.load(f)
    aligned = sum(1 for r in rows if train[r["query_idx"]]["query"] == r["query"])
    print(f"train rows: {len(train):,}; scored pairs whose query_idx matches the train row: {aligned:,} of {len(rows):,}")

    own = own_yes = 0
    for r in rows:
        if r["candidate_text"] in set(train[r["query_idx"]]["negatives"]):
            own += 1
            own_yes += r["llm_answer"] == "Yes"
    other, other_yes = len(rows) - own, sum(r["llm_answer"] == "Yes" for r in rows) - own_yes
    print(f"candidates already in that query's own negatives: {own:,} ({pct(own, len(rows))}), Yes {own_yes:,} ({pct(own_yes, own)})")
    print(f"candidates from other rows: {other:,}, Yes {other_yes:,} ({pct(other_yes, other)})")

    fn_by_query: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.get("is_false_negative"):
            fn_by_query[r["query"]].add(r["candidate_text"])
    touched = removed = 0
    for ex in train:
        fn = fn_by_query.get(ex["query"], set())
        n_removed = sum(1 for neg in ex["negatives"] if neg in fn)
        touched += n_removed > 0
        removed += n_removed
    judged = {r["query_idx"] for r in rows}
    neg_judged = sum(len(train[i]["negatives"]) for i in judged)
    neg_all = sum(len(ex["negatives"]) for ex in train)
    print(f"build_filtered_train.py replay: {removed:,} negatives removed from {touched:,} of {len(train):,} rows "
          f"({pct(touched, len(train))} of rows)")
    print(f"  {pct(removed, neg_judged)} of the {neg_judged:,} negatives in the judged rows, "
          f"{pct(removed, neg_all)} of all {neg_all:,} training negatives")


def draw_figure(dist: Counter, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    ink, ink2, muted, rule, bar = "#0b0b0b", "#52514e", "#6f6e69", "#d9d8d3", "#2a78d6"
    ks = list(range(max(dist) + 1))
    vals = [dist.get(k, 0) for k in ks]
    total = sum(vals)

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(-0.6, len(ks) - 0.4)
    ax.set_ylim(0, max(vals) * 1.22)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.80, bottom=0.20)

    # columns with 4px-style rounded tops, square at the baseline
    width = 0.24
    p0, p1 = ax.transData.transform((0, 0)), ax.transData.transform((1, 1))
    rx, ry = 7 / (p1[0] - p0[0]), 7 / (p1[1] - p0[1])
    c = 0.5523
    for x, h in zip(ks, vals):
        x0, x1 = x - width / 2, x + width / 2
        ryy = min(ry, h / 2)
        verts = [(x0, 0), (x0, h - ryy),
                 (x0, h - ryy + c * ryy), (x0 + rx - c * rx, h), (x0 + rx, h),
                 (x1 - rx, h),
                 (x1 - rx + c * rx, h), (x1, h - ryy + c * ryy), (x1, h - ryy),
                 (x1, 0), (x0, 0)]
        codes = [MPath.MOVETO, MPath.LINETO,
                 MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                 MPath.LINETO,
                 MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                 MPath.LINETO, MPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MPath(verts, codes), facecolor=bar, edgecolor="none"))
        ax.annotate(f"{h:,}", xy=(x, h), xytext=(0, 19), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, color=ink)
        ax.annotate(f"{h / total * 100:.1f}%", xy=(x, h), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=ink2)

    ax.axhline(0, color=rule, linewidth=1)
    ax.set_yticks([])
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], fontsize=10, color=ink)
    ax.tick_params(axis="x", length=0, pad=6)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Candidates judged Yes (flagged as false negatives) per query", fontsize=9.5, color=ink2, labelpad=8)

    with_yes = total - dist.get(0, 0)
    fig.text(0.04, 0.94, "Flagged BM25 candidates per query", fontsize=13, color=ink, ha="left", weight="bold")
    fig.text(0.04, 0.875, f"{total:,} MS MARCO training queries. {with_yes:,} ({with_yes / total * 100:.1f}%) "
             "had at least one candidate judged Yes.", fontsize=9.5, color=ink2, ha="left")
    fig.text(0.04, 0.035, "Each query had up to 5 BM25 candidates; 5 Yes answers were possible only for queries "
             "that kept all 5.", fontsize=8, color=muted, ha="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    print(f"[write] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=DEFAULT_SCORES)
    ap.add_argument("--train", default=None, help="data/train.json from the team's data notebook")
    ap.add_argument("--figure", default=None, help="output PNG path")
    args = ap.parse_args()

    rows = load_scores(Path(args.scores))
    dist = summarize(rows)
    if args.train:
        replay_filter(rows, Path(args.train))
    if args.figure:
        draw_figure(dist, Path(args.figure))


if __name__ == "__main__":
    main()
