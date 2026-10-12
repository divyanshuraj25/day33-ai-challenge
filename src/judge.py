"""LLM-as-judge: scores correctness and groundedness on a 0-3 scale."""
import json
import os
import re

from langchain_google_genai import ChatGoogleGenerativeAI

from .pipeline import message_text

JUDGE_PROMPT = """You are a strict evaluator of a customer-support RAG assistant.

Question: {question}
Reference answer (ground truth): {reference}
Retrieved context given to the assistant:
{context}

Assistant answer: {answer}

Score two things on a 0-3 scale:
- correctness: does the answer agree with the reference answer?
  0 = wrong or contradicts reference, 1 = mostly wrong / key detail missing,
  2 = mostly right with a minor gap, 3 = fully correct.
- groundedness: is EVERY claim in the answer supported by the retrieved context?
  0 = mostly invented, 1 = important claims unsupported, 2 = minor unsupported detail,
  3 = fully supported. An honest "I don't have that information" with no unsupported
  claims is groundedness 3.

Respond with JSON only: {{"correctness": int, "groundedness": int, "reason": "one short sentence"}}"""


class Judge:
    def __init__(self):
        model = os.getenv("JUDGE_MODEL", os.getenv("CHAT_MODEL", "gemini-2.5-flash"))
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0, max_retries=6)

    def score(self, question: str, reference: str, chunks: list, answer: str) -> dict:
        context = "\n".join(f"- {c['text']}" for c in chunks) or "(none)"
        prompt = JUDGE_PROMPT.format(question=question, reference=reference, context=context, answer=answer)
        raw = message_text(self.llm.invoke(prompt))
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)  # tolerate ```json fences
            data = json.loads(match.group(0) if match else raw)
            return {
                "correctness": max(0, min(3, int(data["correctness"]))),
                "groundedness": max(0, min(3, int(data["groundedness"]))),
                "reason": str(data.get("reason", "")),
            }
        except (json.JSONDecodeError, KeyError, ValueError):
            return {"correctness": 0, "groundedness": 0, "reason": f"judge parse error: {raw[:100]}"}
