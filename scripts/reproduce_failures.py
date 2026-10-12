"""Re-run each failing query deliberately with LangSmith tracing.

Each run is tagged `reproduction` + `failure:<id>` and uses session_id `repro-<id>`,
so you can filter for it in the LangSmith dashboard and screenshot the trace.
"""
import json

from src.config import RESULTS_DIR, build_config
from src.pipeline import RagPipeline
from src.tracing import flush


def main():
    failures = json.loads((RESULTS_DIR / "failures.json").read_text(encoding="utf-8"))
    pipe = RagPipeline(build_config())  # baseline config
    out_rows = []
    for f in failures:
        out = pipe.ask(f["question"], session_id=f"repro-{f['id']}",
                       tags=["reproduction", f"failure:{f['id']}"], metadata={"qid": f["id"]})
        print("=" * 80)
        print(f"{f['id']}  run_id={out['run_id']}\nQ: {f['question']}\nA: {out['answer']}")
        for i, c in enumerate(out["chunks"], 1):
            print(f"  chunk {i} score={c['score']} src={c['source']}: {c['text']!r}")
        out_rows.append({"id": f["id"], "run_id": out["run_id"], "answer": out["answer"], "chunks": out["chunks"]})
    (RESULTS_DIR / "reproductions.json").write_text(json.dumps(out_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    flush()
    print("\nOpen LangSmith -> project -> filter tag 'reproduction' and screenshot each trace.")


if __name__ == "__main__":
    main()
