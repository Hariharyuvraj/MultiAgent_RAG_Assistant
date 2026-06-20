import asyncio
import concurrent.futures
import logging
import queue
import re
import threading
from typing import Dict, Generator, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from config import load_config
from guardrails.grounding_check import compute_grounding_score
from ingest.embedder import Embedder
from providers import get_embeddings, get_llm
from schemas.state import AgentState, ChunkResult, WebResult

logger = logging.getLogger(__name__)

_RRF_K = 60  # standard constant for Reciprocal Rank Fusion


def _run_async(coro):
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=60)
    except RuntimeError:
        return asyncio.run(coro)


class RAGPipeline:
    def __init__(self) -> None:
        cfg = load_config()
        self._llm = get_llm(cfg)
        embeddings = get_embeddings(cfg)
        vs_cfg = cfg["vector_store"]
        self._embedder = Embedder(
            embeddings=embeddings,
            persist_path=vs_cfg["persist_path"],
            collection_name=vs_cfg["collection_name"],
        )
        ret_cfg = cfg["retrieval"]
        self._top_k = ret_cfg["top_k"]
        self._threshold = ret_cfg["score_threshold"]
        self._min_relevance = ret_cfg.get("min_relevance_threshold", 0.50)
        self._agent_cfg = cfg["agents"]
        self._guard_cfg = cfg["agents"]["guard"]
        self._vectorstore = None
        self._bm25_index = None
        self._bm25_texts: List[str] = []
        self._bm25_metas: List[dict] = []

    # ── Store management ─────────────────────────────────────────────────────

    def _get_store(self):
        if self._vectorstore is None:
            self._vectorstore = self._embedder.load_store()
        return self._vectorstore

    def reset_store(self) -> None:
        self._vectorstore = None
        self._bm25_index = None
        self._bm25_texts = []
        self._bm25_metas = []

    def delete_from_store(self, filename: str) -> int:
        store = self._get_store()
        results = store._collection.get(where={"source": filename})
        ids = results.get("ids", [])
        if ids:
            store._collection.delete(ids=ids)
        self.reset_store()
        logger.info("Deleted %d chunks for '%s' from vectorstore.", len(ids), filename)
        return len(ids)

    # ── BM25 index ───────────────────────────────────────────────────────────

    def _get_bm25(self) -> Tuple[Optional[object], List[str], List[dict]]:
        if self._bm25_index is not None:
            return self._bm25_index, self._bm25_texts, self._bm25_metas
        try:
            from rank_bm25 import BM25Okapi
            store = self._get_store()
            data = store._collection.get(include=["documents", "metadatas"])
            texts = data.get("documents") or []
            metas = data.get("metadatas") or []
            if not texts:
                return None, [], []
            tokenized = [t.lower().split() for t in texts]
            self._bm25_index = BM25Okapi(tokenized)
            self._bm25_texts = texts
            self._bm25_metas = metas
            logger.info("[BM25] Index built on %d chunks.", len(texts))
            return self._bm25_index, self._bm25_texts, self._bm25_metas
        except Exception:
            logger.warning("[BM25] Index build failed; dense search only.", exc_info=True)
            return None, [], []

    # ── Dense retrieval ──────────────────────────────────────────────────────

    def _dense_search(self, queries: List[str]) -> Dict[Tuple, dict]:
        store = self._get_store()
        ranked: Dict[Tuple, dict] = {}
        global_rank = 0
        for query in queries:
            # similarity_search_with_relevance_scores returns (doc, float) with real scores
            results = store.similarity_search_with_relevance_scores(
                query, k=self._top_k
            )
            for doc, score in results:
                if score < self._threshold:
                    continue
                key = (doc.metadata.get("source", "unknown"), int(doc.metadata.get("page", 0)))
                if key not in ranked:
                    ranked[key] = {
                        "chunk": ChunkResult(
                            content=doc.page_content,
                            source=doc.metadata.get("source", "unknown"),
                            page=int(doc.metadata.get("page", 0)),
                            score=float(score),
                        ),
                        "rank": global_rank,
                    }
                global_rank += 1
        return ranked

    # ── BM25 retrieval ───────────────────────────────────────────────────────

    def _bm25_search(self, queries: List[str]) -> Dict[Tuple, dict]:
        bm25, texts, metas = self._get_bm25()
        if bm25 is None:
            return {}
        ranked: Dict[Tuple, dict] = {}
        for query in queries:
            scores = bm25.get_scores(query.lower().split())
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self._top_k]
            for rank, idx in enumerate(top_indices):
                if scores[idx] <= 0:
                    continue
                meta = metas[idx] if idx < len(metas) else {}
                key = (meta.get("source", "unknown"), int(meta.get("page", 0)))
                if key not in ranked:
                    ranked[key] = {
                        "content": texts[idx],
                        "source": meta.get("source", "unknown"),
                        "page": int(meta.get("page", 0)),
                        "bm25_score": float(scores[idx]),
                        "rank": rank,
                    }
        return ranked

    # ── Reciprocal Rank Fusion ───────────────────────────────────────────────

    def _rrf_fuse(
        self,
        dense: Dict[Tuple, dict],
        bm25: Dict[Tuple, dict],
    ) -> List[ChunkResult]:
        all_keys = set(dense) | set(bm25)
        fused: Dict[Tuple, Tuple[float, ChunkResult]] = {}
        for key in all_keys:
            rrf_score = 0.0
            if key in dense:
                rrf_score += 1.0 / (_RRF_K + dense[key]["rank"] + 1)
            if key in bm25:
                rrf_score += 1.0 / (_RRF_K + bm25[key]["rank"] + 1)

            if key in dense:
                chunk = dense[key]["chunk"]
            else:
                item = bm25[key]
                chunk = ChunkResult(
                    content=item["content"],
                    source=item["source"],
                    page=item["page"],
                    score=item["bm25_score"],
                )
            fused[key] = (rrf_score, chunk)

        sorted_results = sorted(fused.values(), key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in sorted_results[: self._top_k]]

    # ── Web search fallback ──────────────────────────────────────────────────

    def _web_search(self, query: str, max_results: int = 6) -> List[WebResult]:
        try:
            from ddgs import DDGS
            raw = list(DDGS().text(query, max_results=max_results))
            results = [
                WebResult(
                    title=r.get("title", ""),
                    body=r.get("body", ""),
                    url=r.get("href", ""),
                )
                for r in raw
                if r.get("body")
            ]
            logger.info("[WebSearch] Found %d results for: %s", len(results), query[:60])
            return results
        except Exception:
            logger.warning("[WebSearch] ddgs search failed.", exc_info=True)
            return []

    # ── Safety classification ─────────────────────────────────────────────────

    _SAFETY_PROMPT = (
        "You are a strict content safety classifier for a document assistant.\n"
        "Analyse the user query and decide if it requests any of the following:\n"
        "  1. illegal_activity  – instructions for crimes, fraud, hacking, weapon/drug synthesis\n"
        "  2. violence          – planning or glorifying physical harm to people or animals\n"
        "  3. hate_speech       – content targeting race, religion, gender, sexuality, ethnicity\n"
        "  4. self_harm         – methods for suicide or self-injury\n"
        "  5. explicit_content  – sexual content involving minors or non-consensual acts\n"
        "  6. privacy_violation – doxxing, stealing personal data, identity theft\n\n"
        "Legitimate queries about history, law, cybersecurity research, medicine, or education "
        "should be marked SAFE unless they ask for step-by-step harmful instructions.\n\n"
        "Respond ONLY in this exact format (no extra text):\n"
        "VERDICT: SAFE\n"
        "or\n"
        "VERDICT: UNSAFE\n"
        "CATEGORY: <one of the six categories above>"
    )

    def run_safety_check(self, state: AgentState) -> AgentState:
        messages = [
            SystemMessage(content=self._SAFETY_PROMPT),
            HumanMessage(content=f"User query: {state.query}"),
        ]

        async def _call():
            return await self._llm.ainvoke(messages)

        try:
            response = _run_async(_call())
            text = response.content.strip()
            if "VERDICT: UNSAFE" in text:
                category = "policy_violation"
                for line in text.splitlines():
                    if line.startswith("CATEGORY:"):
                        category = line.split(":", 1)[1].strip()
                        break
                state.is_blocked = True
                state.block_category = category
                logger.warning("[SafetyCheck] Blocked query | category=%s | query=%s", category, state.query[:80])
            else:
                state.is_blocked = False
                logger.info("[SafetyCheck] Query passed safety check.")
        except Exception:
            logger.warning("[SafetyCheck] Classification failed — allowing query.", exc_info=True)
            state.is_blocked = False
        return state

    # ── Agent steps ──────────────────────────────────────────────────────────

    def run_context(self, state: AgentState) -> AgentState:
        if not state.history:
            state.enriched_query = state.query
            return state

        max_turns = self._agent_cfg["context"]["max_history_turns"]
        recent = state.history[-max_turns:]
        formatted = "\n".join(
            f"User: {t['user']}\nAssistant: {t['assistant']}" for t in recent
        )
        messages = [
            SystemMessage(
                content=(
                    "You are a conversation analyst. Answer ONLY 'YES' if the new "
                    "query is a follow-up to the conversation, or 'NO' if it is a fresh topic."
                )
            ),
            HumanMessage(content=f"Conversation:\n{formatted}\n\nNew query: {state.query}"),
        ]

        async def _call():
            return await self._llm.ainvoke(messages)

        response = _run_async(_call())
        if response.content.strip().upper().startswith("YES"):
            state.enriched_query = (
                f"Context from prior conversation:\n{formatted}\n\n"
                f"Current question: {state.query}"
            )
        else:
            state.enriched_query = state.query
        logger.info("[ContextAgent] enriched_query set.")
        return state

    def run_planner(self, state: AgentState) -> AgentState:
        system = (
            "You are a query planning expert. Break the user question into 2 to 4 "
            "specific retrieval sub-steps. Return ONLY a numbered list, one per line."
        )
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=state.enriched_query),
        ]

        async def _call():
            return await self._llm.ainvoke(messages)

        response = _run_async(_call())
        plan = []
        for line in response.content.strip().splitlines():
            cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
            if cleaned:
                plan.append(cleaned)

        max_steps = self._agent_cfg["planner"]["max_steps"]
        state.plan = (plan or [state.enriched_query])[:max_steps]
        logger.info("[PlannerAgent] Plan: %s", state.plan)
        return state

    # Phrases that signal the user wants real-time / current information.
    # Documents are static — these queries must always go to web search.
    _TEMPORAL_PATTERNS = (
        "latest", "current", "currently", "recent", "recently",
        "right now", "today", "this year", "this month", "this week",
        "who is the", "who is cm", "who is pm", "who is president",
        "who is ceo", "who is minister", "who is governor", "who is chief",
        "who won", "who leads", "who runs",
        "news", "update", "updates", "breaking", "live",
        "at present", "presently", "ongoing", "happening", "now",
        "2024", "2025", "2026",
    )

    def _is_temporal_query(self, query: str) -> bool:
        q = query.lower()
        return any(p in q for p in self._TEMPORAL_PATTERNS)

    def run_search(self, state: AgentState) -> AgentState:
        # ── Real-time gate: bypass documents for time-sensitive queries ──
        if self._is_temporal_query(state.query):
            logger.info("[SearchAgent] Temporal query detected — routing directly to web search.")
            web_results = self._web_search(state.enriched_query)
            state.web_sources = web_results
            state.is_web_answer = True
            state.retrieved_chunks = []
            return state

        # ── Hybrid retrieval: dense vector + BM25 → RRF fusion ──
        dense = self._dense_search(state.plan)
        bm25  = self._bm25_search(state.plan)
        fused = self._rrf_fuse(dense, bm25)

        best_dense_score = (
            max(v["chunk"].score for v in dense.values()) if dense else 0.0
        )

        logger.info(
            "[SearchAgent] Hybrid results: %d fused (dense=%d bm25=%d) best_score=%.3f threshold=%.3f",
            len(fused), len(dense), len(bm25), best_dense_score, self._min_relevance,
        )

        # Only trust doc results when at least one chunk has high semantic relevance.
        if dense and best_dense_score >= self._min_relevance:
            state.retrieved_chunks = fused
            state.is_web_answer = False
            return state

        reason = (
            f"best dense score {best_dense_score:.3f} < {self._min_relevance}"
            if dense else "no dense matches"
        )
        logger.info("[SearchAgent] Out-of-domain (%s) — falling back to web search.", reason)
        web_results = self._web_search(state.enriched_query)
        state.web_sources = web_results
        state.is_web_answer = True
        state.retrieved_chunks = []
        return state

    # ── Answer streaming ─────────────────────────────────────────────────────

    def stream_answer(self, state: AgentState) -> Generator[str, None, None]:
        if state.is_web_answer:
            yield from self._stream_web_answer(state)
            return

        if not state.retrieved_chunks:
            msg = (
                "I couldn't find anything relevant in the documents for that question. "
                "Try rephrasing, or upload a document that covers this topic."
            )
            state.answer = msg
            state.sources = []
            yield msg
            return

        yield from self._stream_doc_answer(state)

    def _stream_doc_answer(self, state: AgentState) -> Generator[str, None, None]:
        parts = [
            f"[{i}] Source: {c.source}, Page: {c.page}\n{c.content}"
            for i, c in enumerate(state.retrieved_chunks, 1)
        ]
        context_block = "\n\n".join(parts)
        messages = [
            SystemMessage(
                content=(
                    "You are a knowledgeable document assistant. Answer thoroughly and helpfully.\n"
                    "1. Start with a clear, direct answer.\n"
                    "2. Expand with relevant details, context, and explanation from the documents.\n"
                    "3. Use bullet points or numbered lists for multi-part answers, steps, or comparisons.\n"
                    "4. Cite sources inline as [filename:page] where relevant.\n"
                    "5. Write as much as needed to fully answer the question — do not cut short.\n"
                    "6. Never repeat the question or add filler phrases like 'Great question!'.\n"
                    "7. If the answer is not in the documents, say so clearly in one sentence."
                )
            ),
            HumanMessage(
                content=f"Question: {state.enriched_query}\n\nDocument Excerpts:\n{context_block}"
            ),
        ]

        token_queue: queue.Queue = queue.Queue()

        async def _stream():
            async for chunk in self._llm.astream(messages):
                if chunk.content:
                    token_queue.put(chunk.content)
            token_queue.put(None)

        threading.Thread(target=lambda: asyncio.run(_stream()), daemon=True).start()

        full_answer = ""
        while True:
            token = token_queue.get(timeout=60)
            if token is None:
                break
            full_answer += token
            yield token

        state.answer = full_answer
        cited = set()
        for chunk in state.retrieved_chunks:
            if re.search(re.escape(chunk.source), full_answer, re.IGNORECASE):
                cited.add(f"{chunk.source}:{chunk.page}")
        state.sources = sorted(cited) if cited else [
            f"{c.source}:{c.page}" for c in state.retrieved_chunks
        ]

    def _stream_web_answer(self, state: AgentState) -> Generator[str, None, None]:
        if not state.web_sources:
            # Web search returned nothing — answer from LLM's own training knowledge
            logger.info("[WebSearch] No web results — answering from LLM knowledge.")
            yield from self._stream_from_llm_knowledge(state)
            return

        context_block = "\n\n".join(
            f"[{i}] {r.title}\n{r.body}\nURL: {r.url}"
            for i, r in enumerate(state.web_sources, 1)
        )
        messages = [
            SystemMessage(
                content=(
                    "You are a helpful assistant answering questions using live web search results.\n"
                    "1. Give a clear, direct answer first.\n"
                    "2. Provide full context — explain the background, key facts, timeline, and significance.\n"
                    "3. Use bullet points or numbered lists for events, comparisons, or multi-part facts.\n"
                    "4. Cite sources as [1], [2], etc. after each fact that comes from that source.\n"
                    "5. Write a complete, thorough response — do not truncate the answer.\n"
                    "6. Never fabricate facts not present in the search results.\n"
                    "7. End with a brief summary if the answer is complex."
                )
            ),
            HumanMessage(
                content=f"Question: {state.query}\n\nWeb Search Results:\n{context_block}"
            ),
        ]

        token_queue: queue.Queue = queue.Queue()

        async def _stream():
            async for chunk in self._llm.astream(messages):
                if chunk.content:
                    token_queue.put(chunk.content)
            token_queue.put(None)

        threading.Thread(target=lambda: asyncio.run(_stream()), daemon=True).start()

        full_answer = ""
        while True:
            token = token_queue.get(timeout=60)
            if token is None:
                break
            full_answer += token
            yield token

        state.answer = full_answer
        state.sources = [r.url for r in state.web_sources]

    def _stream_from_llm_knowledge(self, state: AgentState) -> Generator[str, None, None]:
        messages = [
            SystemMessage(
                content=(
                    "You are a knowledgeable general assistant. Answer the user's question "
                    "using your training knowledge.\n"
                    "1. Give a clear, direct answer first.\n"
                    "2. Provide full context, background, and relevant details.\n"
                    "3. Use bullet points or numbered lists where appropriate.\n"
                    "4. If your knowledge has a cutoff and the question requires very recent info, "
                    "say so briefly but still answer with what you know.\n"
                    "5. Never refuse to answer a legitimate question."
                )
            ),
            HumanMessage(content=state.query),
        ]

        token_queue: queue.Queue = queue.Queue()

        async def _stream():
            async for chunk in self._llm.astream(messages):
                if chunk.content:
                    token_queue.put(chunk.content)
            token_queue.put(None)

        threading.Thread(target=lambda: asyncio.run(_stream()), daemon=True).start()

        full_answer = ""
        while True:
            token = token_queue.get(timeout=60)
            if token is None:
                break
            full_answer += token
            yield token

        state.answer = full_answer
        state.sources = []

    # ── Guard ─────────────────────────────────────────────────────────────────

    # ── Non-streaming full pipeline run (used by RAGAS eval) ─────────────────

    def run_sync(self, question: str, history: Optional[List[dict]] = None) -> AgentState:
        """Run the full pipeline synchronously and return the final AgentState.
        Collects the complete answer in memory instead of streaming tokens.
        """
        import uuid
        state = AgentState(
            query=question,
            session_id=str(uuid.uuid4())[:8],
            history=history or [],
        )

        state = self.run_safety_check(state)
        if state.is_blocked:
            state.answer = f"[Blocked: {state.block_category}]"
            state.passed_guard = False
            return state

        state = self.run_context(state)
        state = self.run_planner(state)
        state = self.run_search(state)

        # collect full answer without streaming
        full_answer = "".join(self.stream_answer(state))
        state.answer = full_answer

        state = self.run_guard(state)
        return state

    def run_guard(self, state: AgentState) -> AgentState:
        # Web answers are not grounded against local docs — skip the check
        if state.is_web_answer:
            state.passed_guard = True
            state.grounding_score = 1.0
            return state

        score = compute_grounding_score(state.answer, state.retrieved_chunks)
        state.grounding_score = score
        threshold = float(self._guard_cfg["grounding_threshold"])

        if score >= threshold:
            state.passed_guard = True
        else:
            state.passed_guard = False
            state.answer = self._guard_cfg["fallback_message"]
            state.sources = []

        logger.info(
            "[GuardAgent] score=%.3f threshold=%.3f passed=%s",
            score, threshold, state.passed_guard,
        )
        return state
