# Day 30 - AI System Debugging with LangSmith

A small RAG pipeline (NimbusDesk support bot) with full LangSmith tracing, structured JSON logging,
component isolation tests, an eval suite with an LLM judge, and a debugging runbook.

```
retrieve -> build_prompt -> generate
   |             |             |
   +-- each step = LangSmith span + JSON log line (timestamp, session_id, step_name,
                   input_summary, output_summary, latency_ms)
```

## Setup (Windows / VS Code)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then put GOOGLE_API_KEY (Gemini) and LANGSMITH_API_KEY in .env
```

## Run order (this produces every submission artifact)

| # | Command | Produces |
|---|---|---|
| 1 | `python -m scripts.run_eval --label baseline` | `results/eval_baseline.json` (scores per query) |
| 2 | `python -m scripts.find_failures` | `results/failures.json` (3 queries scoring < 2) |
| 3 | `python -m scripts.reproduce_failures` | 3 traces in LangSmith (tag `reproduction`) |
| 4 | `python -m scripts.isolation_tests` | per-stage input/output + root-cause diagnosis |
| 5 | `python -m scripts.run_eval --fixes chunking,prompt --label after` | scores after the two fixes |
| 6 | `python -m scripts.compare_results --before baseline --after after` | `results/summary.md` (before/after table) |

Step 4 prints the diagnosis for each failure. Choose the two fixes that match your diagnoses
(`chunking`, `prompt`, `threshold`) and pass them in step 5. If a diagnosis points elsewhere,
change `FIXES` in `src/config.py`.

## What to submit

- **LangSmith traces for three reproduced failures** - screenshots from the LangSmith dashboard (filter tag `reproduction`), plus the `run_id`s in `results/reproductions.json`.
- **Component isolation test scripts** - `scripts/isolation_tests.py`
- **Structured JSON logging implementation** - `src/json_logging.py` and a sample `logs/pipeline.jsonl`
- **Two fixed failures with before/after eval scores** - `results/summary.md`
- **Debugging runbook** - `RUNBOOK.md`

## Using your own Day 29 eval set

Replace `eval/dataset.json` with your questions. Required fields: `id`, `question`, `reference_answer`,
`category`. For isolation tests also add `key_fact` and `source_doc` (or `"expects_abstain": true`
when no answer exists in the docs). If you have your own pipeline, keep the same three step methods.

## Notes

- Scores are 0-3; a query **fails** when correctness or groundedness is below 2 (`FAIL_THRESHOLD` in `src/config.py`).
- Uses Google Gemini (chat + embeddings) via `langchain-google-genai`; change models in `.env` if needed.
- The baseline is deliberately weak (60-character chunks, permissive prompt) so real failures exist to debug.
- `threshold` fix: `min_score=0.25` is a starting point; tune it by looking at `top_score` in the logs.
- LLM-judge scores can vary slightly between runs; the scripts use temperature 0 to keep this small.

## Push to GitHub

```powershell
git init
git add .
git commit -m "Day 30: AI debugging workflow with LangSmith"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
`.env` is git-ignored, so your API keys stay private.
