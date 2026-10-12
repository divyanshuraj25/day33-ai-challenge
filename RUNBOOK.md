# AI Debugging Runbook

Follow these steps **in order** every time the AI system returns a wrong answer.
The system runs without errors and still gives wrong answers, so do not start with stack traces. Start with the trace.

1. **Capture the failure.** Save the exact question, the wrong answer, the `session_id` and the `run_id`. Add the query to `eval/dataset.json` with a reference answer and `key_fact` so it can never regress silently.
2. **Reproduce it deliberately.** Re-run only that query with tracing on (`python -m scripts.reproduce_failures`). If it does not fail again, check `temperature`, config and data version before going further.
3. **Open the LangSmith trace.** Filter by tag `failure:<id>` or by `run_id`. Read the three spans in order: `retrieve` -> `build_prompt` -> `generate`. Look at the *input and output of each span*, not only the final answer.
4. **Check the retrieved chunks first.** Does any chunk contain the key fact? Are the similarity scores low? Is the fact cut in half across two chunks? If the fact is missing, this is a **retrieval or chunking bug**. Stop here and fix upstream; a better prompt cannot recover information that was never retrieved.
5. **Check the built prompt.** If the chunks were right, confirm the context actually reached the system prompt and that the wording tells the model to rely on it. If the prompt allows "general knowledge" or "industry norms", the model will drift. This is a **prompt bug**.
6. **Check generation with gold context.** Run `python -m scripts.isolation_tests`. Stage 3a feeds perfect context to the model. If it still answers wrongly, the problem is **generation or prompt wording**. If it answers correctly, the problem is upstream (steps 4-5).
7. **Check the "no answer exists" case.** If the docs do not contain the answer, the correct behaviour is to say so. A confident answer here is a **hallucination**. Fix with a strict "answer only from context / say you don't know" instruction and, if needed, a retrieval score threshold (`min_score`).
8. **Read the structured logs.** `logs/pipeline.jsonl` has `timestamp, session_id, step_name, input_summary, output_summary, latency_ms` per step. Filter by `session_id` to follow one request. Look for a suspicious `top_score`, an empty chunk list, or an unusually slow step.
9. **Change ONE thing at a time.** Pick the single knob matching the root cause: chunk size/overlap, system prompt wording, or retrieval threshold. Changing several at once makes it impossible to know what worked.
10. **Re-run the whole eval suite** (`python -m scripts.run_eval --fixes ... --label after`). Confirm the failing query now scores >= 2 **and** that no other query regressed (`python -m scripts.compare_results`).
11. **Lock it in.** Keep the query in the eval set, commit the config change with the before/after scores in the commit message, and note the root cause category (retrieval / prompt / generation / hallucination) so patterns become visible over time.

## Quick root-cause table

| Symptom in the trace | Root cause | Fix |
|---|---|---|
| Key fact missing from retrieved chunks, or split across chunks | Chunking / retrieval | Larger chunks, overlap, higher `top_k` |
| Fact in the prompt but answer contradicts it | Prompt ignored context | Rewrite system prompt: "answer ONLY from context" |
| Correct with gold context, wrong with retrieved context | Retrieval | Fix retrieval, not generation |
| Wrong even with gold context | Generation drift | Tighten prompt, lower temperature, shorter answers |
| Confident answer when docs are silent | Hallucination | "Say you don't know" instruction + `min_score` threshold |
