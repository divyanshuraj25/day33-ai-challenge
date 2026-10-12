"""Run the eval suite and save results/eval_<label>.json.

  python -m scripts.run_eval --label baseline
  python -m scripts.run_eval --fixes chunking,prompt --label after
"""
import argparse
import json
import os
import time
from statistics import mean

from src.config import DATASET_PATH, FAIL_THRESHOLD, RESULTS_DIR, build_config
from src.judge import Judge
from src.pipeline import RagPipeline
from src.tracing import flush


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixes", default="", help="comma list: chunking,prompt,threshold")
    ap.add_argument("--label", default=None, help="output name (default: config name)")
    args = ap.parse_args()

    fixes = [f for f in args.fixes.split(",") if f]
    cfg = build_config(fixes)
    label = args.label or cfg.name
    pipe, judge = RagPipeline(cfg), Judge()
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    records = []
    for item in dataset:
        out = pipe.ask(item["question"], session_id=f"eval-{label}-{item['id']}",
                       tags=[f"eval:{label}", f"qid:{item['id']}"], metadata={"qid": item["id"]})
        s = judge.score(item["question"], item["reference_answer"], out["chunks"], out["answer"])
        records.append({**item, "answer": out["answer"], "chunks": out["chunks"],
                        "run_id": out["run_id"], **s})
        time.sleep(float(os.getenv("CALL_DELAY", "2")))  # stay under free-tier rate limits
        flag = "FAIL" if min(s["correctness"], s["groundedness"]) < FAIL_THRESHOLD else "ok  "
        print(f"{flag} {item['id']} corr={s['correctness']} grnd={s['groundedness']} | {item['question']}")

    summary = {
        "avg_correctness": round(mean(r["correctness"] for r in records), 3),
        "avg_groundedness": round(mean(r["groundedness"] for r in records), 3),
        "num_failures": sum(1 for r in records if min(r["correctness"], r["groundedness"]) < FAIL_THRESHOLD),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"eval_{label}.json"
    path.write_text(json.dumps({"label": label, "config": cfg.name, "summary": summary,
                                "records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSUMMARY", summary, "->", path)
    flush()


if __name__ == "__main__":
    main()
