"""RAG pipeline: retrieve -> build_prompt -> generate.

Every step is
  * a plain method (so it can be run ALONE in the isolation tests),
  * wrapped in a RunnableLambda (so it shows up as its own span in LangSmith),
  * wrapped in log_step (structured JSON logging).
"""
import uuid
from typing import Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import DOCS_DIR, PipelineConfig
from .json_logging import log_step, summarize
from .tracing import get_tracer

NO_CONTEXT = "(no relevant context retrieved)"


def message_text(response) -> str:
    """Return plain text from a chat response (content may be a str or a list of blocks)."""
    content = response.content
    if isinstance(content, str):
        return content.strip()
    parts = [b if isinstance(b, str) else b.get("text", "") for b in content
             if isinstance(b, str) or (isinstance(b, dict) and b.get("type") == "text")]
    return "".join(parts).strip()


def load_documents() -> list[Document]:
    return [
        Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
        for p in sorted(DOCS_DIR.glob("*.txt"))
    ]


def read_gold_doc(source_doc: str) -> str:
    """Full text of the source document that contains the answer (for isolation tests)."""
    return (DOCS_DIR / source_doc).read_text(encoding="utf-8")


class RagPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.embeddings = GoogleGenerativeAIEmbeddings(model=config.embedding_model)
        self.llm = ChatGoogleGenerativeAI(model=config.chat_model, temperature=0, max_retries=6)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )
        self.chunks = splitter.split_documents(load_documents())
        self.store = InMemoryVectorStore(embedding=self.embeddings)
        self.store.add_documents(self.chunks)

        self.chain = (
            RunnableLambda(self.retrieve).with_config(run_name="retrieve")
            | RunnableLambda(self.build_prompt).with_config(run_name="build_prompt")
            | RunnableLambda(self.generate).with_config(run_name="generate")
        )

    # ------------------------------------------------------------------ steps
    def retrieve(self, state: dict) -> dict:
        question, sid = state["question"], state.get("session_id", "na")
        with log_step(sid, "retrieve", question) as rec:
            results = self.store.similarity_search_with_score(question, k=self.config.top_k)
            chunks = [
                {"text": d.page_content, "score": round(float(s), 4), "source": d.metadata["source"]}
                for d, s in results
                if self.config.min_score is None or s >= self.config.min_score
            ]
            top = chunks[0]["score"] if chunks else None
            rec["output_summary"] = (
                f"{len(chunks)}/{len(results)} chunks kept, top_score={top}, "
                f"first_chunk={summarize(chunks[0]['text'], 80) if chunks else None}"
            )
        return {**state, "chunks": chunks}

    def build_prompt(self, state: dict) -> dict:
        sid = state.get("session_id", "na")
        with log_step(sid, "build_prompt", f"{len(state['chunks'])} chunks") as rec:
            context = (
                "\n\n".join(
                    f"[{i}] ({c['source']}) {c['text']}" for i, c in enumerate(state["chunks"], 1)
                )
                or NO_CONTEXT
            )
            system = self.config.system_prompt.replace("{context}", context)
            messages = [SystemMessage(content=system), HumanMessage(content=state["question"])]
            rec["output_summary"] = f"system_prompt_chars={len(system)}, context_chars={len(context)}"
        return {**state, "messages": messages}

    def generate(self, state: dict, config: Optional[RunnableConfig] = None) -> dict:
        sid = state.get("session_id", "na")
        with log_step(sid, "generate", state["question"]) as rec:
            response = self.llm.invoke(state["messages"], config=config)
            answer = message_text(response)
            rec["output_summary"] = answer
        return {**state, "answer": answer}

    # ------------------------------------------------------------- public API
    def ask(
        self,
        question: str,
        session_id: Optional[str] = None,
        tags: Optional[list] = None,
        metadata: Optional[dict] = None,
        trace: bool = True,
    ) -> dict:
        """Run the full pipeline. Returns the final state plus session_id and run_id."""
        session_id = session_id or uuid.uuid4().hex[:8]
        run_id = uuid.uuid4()
        tracer = get_tracer() if trace else None
        result = self.chain.invoke(
            {"question": question, "session_id": session_id},
            config={
                "run_id": run_id,
                "run_name": "rag_pipeline",
                "callbacks": [tracer] if tracer else [],  # <- LangSmith callback
                "tags": [self.config.name, *(tags or [])],
                "metadata": {"session_id": session_id, "config": self.config.name, **(metadata or {})},
            },
        )
        result["run_id"] = str(run_id)
        return result
