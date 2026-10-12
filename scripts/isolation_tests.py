"""Component isolation tests: run retrieval, prompt construction and generation ALONE.

  python -m scripts.isolation_tests
  python -m scripts.isolation_tests --fixes chunking,prompt   # re-check after a fix
"""
import argparse
import json

from src.config import RESULTS_DIR, build_config
from src.pipeline import RagPipeline, read_gold_doc

ABSTAIN = ("don't have", "do not have", "not mentioned", "no information", "not available", "not listed")


def has(text: str, fact) -> bool:
    return fact is not None and fact.lower() in text.lower()


def abstained(text: str) -> bool:
    return any(m in text.lower() for m in ABSTAIN)


def banner(t):
    print(f"\n--- {t} " + "-" * max(0, 70 - len(t)))


def test_one(pipe: RagPipeline, f: dict) -> dict:
    q, fact, sid = f["question"], f.get("key_fact"), f"iso-{f['id']}"
    abstain_case = f.get("expects_abstain", False)
    print("=" * 80, f"\n{f['id']} [{f['category']}]  Q: {q}\n   expected key fact: {fact!r}")

    # STAGE 1 - retrieval alone
    banner("STAGE 1: retrieval alone")
    s1 = pipe.retrieve({"question": q, "session_id": sid})
    print("INPUT :", q)
    for i, c in enumerate(s1["chunks"], 1):
        print(f"OUTPUT chunk {i} score={c['score']} src={c['source']}: {c['text']!r}")
    if not s1["chunks"]:
        print("OUTPUT: (no chunks)")
    retrieval_ok = None if abstain_case else any(has(c["text"], fact) for c in s1["chunks"])
    print("CHECK : retrieved chunks contain key fact ->", retrieval_ok if retrieval_ok is not None else "n/a (out of scope)")

    # STAGE 2 - prompt construction alone
    banner("STAGE 2: prompt construction alone")
    s2 = pipe.build_prompt(s1)
    system_text = s2["messages"][0].content
    print("INPUT :", f"{len(s1['chunks'])} chunks + question")
    print("OUTPUT system prompt:\n" + system_text)
    prompt_ok = None if abstain_case else has(system_text, fact)
    print("CHECK : prompt contains key fact ->", prompt_ok if prompt_ok is not None else "n/a")

    # STAGE 3a - generation alone, with PERFECT (gold) context
    banner("STAGE 3a: generation alone, gold context")
    if abstain_case:
        gold_chunks = []
    else:
        gold_chunks = [{"text": read_gold_doc(f["source_doc"]), "score": 1.0, "source": f["source_doc"]}]
    gold_state = pipe.build_prompt({"question": q, "session_id": sid, "chunks": gold_chunks})
    gold_ans = pipe.generate(gold_state)["answer"]
    print("OUTPUT:", gold_ans)
    gold_ok = abstained(gold_ans) if abstain_case else has(gold_ans, fact)
    print("CHECK : answer correct with perfect context ->", gold_ok)

    # STAGE 3b - generation alone, with the RETRIEVED prompt
    banner("STAGE 3b: generation alone, retrieved prompt")
    real_ans = pipe.generate(s2)["answer"]
    print("OUTPUT:", real_ans)
    real_ok = abstained(real_ans) if abstain_case else has(real_ans, fact)
    print("CHECK : answer correct with retrieved context ->", real_ok)

    # DIAGNOSIS
    if real_ok:
        diag = "NOT REPRODUCED (answer is correct now)"
    elif abstain_case:
        top = s1["chunks"][0]["score"] if s1["chunks"] else None
        diag = (f"HALLUCINATION: no answer exists in the docs, yet the model answered anyway "
                f"(top retrieval score={top}). Fix: stricter prompt and/or retrieval threshold.")
    elif not retrieval_ok:
        diag = ("RETRIEVAL: the key fact is not in any retrieved chunk "
                + ("(model answers correctly with gold context, so generation is fine). Fix: chunking."
                   if gold_ok else "(and generation also struggles)."))
    elif not gold_ok:
        diag = "GENERATION: model drifts from facts even with perfect context. Fix: system prompt wording."
    else:
        diag = "PROMPT: fact was in the prompt but the answer ignored it. Fix: system prompt wording."
    print("\nDIAGNOSIS:", diag)
    return {"id": f["id"], "retrieval_ok": retrieval_ok, "prompt_ok": prompt_ok,
            "gold_generation_ok": gold_ok, "retrieved_generation_ok": real_ok, "diagnosis": diag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixes", default="")
    args = ap.parse_args()
    cfg = build_config([f for f in args.fixes.split(",") if f])
    pipe = RagPipeline(cfg)
    failures = json.loads((RESULTS_DIR / "failures.json").read_text(encoding="utf-8"))
    report = [test_one(pipe, f) for f in failures]
    path = RESULTS_DIR / f"isolation_report_{cfg.name}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved ->", path)


if __name__ == "__main__":
    main()
