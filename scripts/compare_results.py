"""Before/after comparison -> results/summary.md (paste into your submission).

  python -m scripts.compare_results --before baseline --after after
"""
import argparse
import json

from src.config import FAIL_THRESHOLD, RESULTS_DIR


def load(label):
    return json.loads((RESULTS_DIR / f"eval_{label}.json").read_text(encoding="utf-8"))


def worst(r):
    return min(r["correctness"], r["groundedness"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="baseline")
    ap.add_argument("--after", default="after")
    args = ap.parse_args()

    before, after = load(args.before), load(args.after)
    b = {r["id"]: r for r in before["records"]}
    a = {r["id"]: r for r in after["records"]}
    failures = json.loads((RESULTS_DIR / "failures.json").read_text(encoding="utf-8"))

    L = [f"# Eval results: `{before['config']}` -> `{after['config']}`", "",
         "## Overall", "", "| Metric | Before | After |", "|---|---|---|"]
    for k in ("avg_correctness", "avg_groundedness", "num_failures"):
        L.append(f"| {k} | {before['summary'][k]} | {after['summary'][k]} |")

    L += ["", "## The three reproduced failures", "",
          "| ID | Question | Correctness (before -> after) | Groundedness (before -> after) | Status |",
          "|---|---|---|---|---|"]
    for f in failures:
        x, y = b[f["id"]], a[f["id"]]
        status = "FIXED" if worst(y) >= FAIL_THRESHOLD else "still failing"
        L.append(f"| {f['id']} | {f['question']} | {x['correctness']} -> {y['correctness']} | "
                 f"{x['groundedness']} -> {y['groundedness']} | {status} |")

    reg = [i for i in b if worst(a[i]) < worst(b[i])]
    L += ["", "## Regressions (queries whose score dropped)", ""]
    L += [f"- {i}: {worst(b[i])} -> {worst(a[i])} ({a[i]['question']})" for i in reg] or ["- None"]
    text = "\n".join(L) + "\n"
    (RESULTS_DIR / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
