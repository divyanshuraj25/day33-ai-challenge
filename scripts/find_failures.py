"""Pick the 3 worst failing queries (score < FAIL_THRESHOLD) from the baseline eval.

  python -m scripts.find_failures                 # auto-pick
  python -m scripts.find_failures --ids q01,q06,q13   # or choose manually
"""
import argparse
import json

from src.config import FAIL_THRESHOLD, RESULTS_DIR


def worst(r):
    return min(r["correctness"], r["groundedness"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="eval_baseline.json")
    ap.add_argument("--ids", default="")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    records = json.loads((RESULTS_DIR / args.input).read_text(encoding="utf-8"))["records"]
    if args.ids:
        wanted = args.ids.split(",")
        picked = [r for r in records if r["id"] in wanted]
    else:
        failing = sorted((r for r in records if worst(r) < FAIL_THRESHOLD),
                         key=lambda r: (worst(r), r["correctness"]))
        picked = failing[: args.n]
        if len(failing) < args.n:
            print(f"WARNING: only {len(failing)} queries scored < {FAIL_THRESHOLD}. "
                  f"Add harder questions to eval/dataset.json or use --ids.")

    (RESULTS_DIR / "failures.json").write_text(json.dumps(picked, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in picked:
        print(f"{r['id']} [{r['category']}] corr={r['correctness']} grnd={r['groundedness']}")
        print(f"   Q: {r['question']}\n   A: {r['answer']}\n   judge: {r['reason']}\n")
    print(f"Saved {len(picked)} failures -> results/failures.json")


if __name__ == "__main__":
    main()
