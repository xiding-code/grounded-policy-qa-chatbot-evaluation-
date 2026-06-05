"""End-to-end entrypoint with full memory + adaptive retrieval stack.

Layers (executed in order per turn):
  1. L3 Entity tracking      — extract & persist (order_id, product, reason, intent)
  2. Self-RAG router         — decide whether to retrieve at all
  3. Hybrid retrieval        — BM25 + FAISS via RRF + cross-encoder rerank
  4. CRAG quality judge      — score retrieval; web fallback if LOW
  5. L2 summary buffer       — compress older turns if buffer too long
  6. L1 message injection    — recent turns + summary go to LLM
  7. Structured output       — APPROVE/DENY/ESCALATE/ABSTAIN + safety guard

If a real API key is configured in .env, calls Claude or OpenAI per LLM_PROVIDER.
If not, falls back to deterministic heuristic so the pipeline still runs.
"""
from __future__ import annotations
from src.retrieve import HybridRetriever
from src.schema import build_user_prompt, heuristic_decide, parse_structured, SYSTEM_PROMPT
from src.llm import call_llm, is_configured, LLMNotConfigured
from src.memory import SessionManager
from src.entity_tracker import extract as extract_entities, EntityStore
from src.self_rag import route as self_rag_route, answer_skip
from src.summary_buffer import compress as compress_buffer
from src.crag import maybe_correct


SUMMARY_TRIGGER_TURNS = 5     # start summarizing when raw turns > this
KEEP_RAW_TURNS_AFTER = 3      # always keep this many recent turns verbatim


class AmazonRAGChatbot:
    def __init__(
        self,
        use_reranker: bool = True,
        force_heuristic: bool = False,
        max_turns: int = 12,
        enable_self_rag: bool = True,
        enable_crag: bool = True,
        enable_entity_tracker: bool = True,
        enable_summary: bool = True,
    ):
        self.retriever = HybridRetriever(use_reranker=use_reranker)
        self.sessions = SessionManager(max_turns=max_turns)
        self.entities = EntityStore()
        self.session_summaries: dict[str, str] = {}

        self.enable_self_rag = enable_self_rag
        self.enable_crag = enable_crag
        self.enable_entity_tracker = enable_entity_tracker
        self.enable_summary = enable_summary

        ok, info = is_configured()
        self.use_llm = ok and not force_heuristic
        self.provider_info = info
        if not ok:
            print(f"[chat] LLM not configured ({info}); using heuristic stub.")
        else:
            features = []
            if enable_self_rag:        features.append("Self-RAG")
            if enable_crag:            features.append("CRAG")
            if enable_entity_tracker:  features.append("entity-tracker")
            if enable_summary:         features.append("summary-buffer")
            features.append(f"L1-buffer({max_turns})")
            print(f"[chat] LLM ready: {info} | layers: {' + '.join(features)}")

    def ask(
        self,
        question: str,
        session_id: str | None = None,
        k_candidates: int = 20,
        k_final: int = 5,
    ) -> dict:
        # ===== Layer L3: entity extraction + persistence =====
        entities_updated: list[str] = []
        entities_snapshot: dict = {}
        if self.enable_entity_tracker and session_id:
            new_e = extract_entities(question, use_llm=self.use_llm)
            entities_updated = self.entities.update(session_id, new_e)
            entities_snapshot = self.entities.get(session_id).to_dict()

        # ===== Layer: Self-RAG router =====
        if self.enable_self_rag:
            route_decision = self_rag_route(question)
        else:
            from src.self_rag import RouteDecision
            route_decision = RouteDecision(route="RETRIEVE", reason="disabled", used_llm=False)

        # If SKIP, generate a direct response and return early
        if route_decision.route == "SKIP":
            answer = answer_skip(question)
            self._persist_turn(session_id, question, answer)
            return self._build_response(
                question=question,
                session_id=session_id,
                decision={
                    "decision": "ABSTAIN",
                    "answer": answer,
                    "required_info": "NONE",
                    "citations": "NONE",
                    "raw": answer,
                    "guarded": False,
                },
                evidence=[],
                n_prior_turns=self._n_prior_turns(session_id) - 1,  # before this turn
                self_rag=route_decision,
                crag=None,
                summary_info=None,
                entities=entities_snapshot,
                entities_updated=entities_updated,
                skipped_retrieval=True,
            )

        # ===== Layer: Hybrid retrieval + rerank =====
        hits = self.retriever.retrieve(question, candidates=k_candidates, final_k=k_final)
        evidence_dicts = [self._hit_to_dict(h) for h in hits]

        # ===== Layer: CRAG quality judge + optional web fallback =====
        crag_verdict = None
        if self.enable_crag and hits:
            crag_verdict = maybe_correct(question, evidence_dicts, enable_web_fallback=True)
            if crag_verdict.quality == "LOW" and crag_verdict.fallback:
                evidence_dicts = evidence_dicts + crag_verdict.fallback

        # ===== Layer L1 + L2: history with optional summary =====
        history: list[dict] = []
        n_prior_turns = 0
        summary_info = None
        if session_id:
            buffer = self.sessions.get(session_id)
            raw_history = buffer.messages()
            n_prior_turns = buffer.n_turns()

            if self.enable_summary and n_prior_turns > SUMMARY_TRIGGER_TURNS:
                summarized = compress_buffer(
                    raw_history,
                    keep_raw_turns=KEEP_RAW_TURNS_AFTER,
                    prior_summary=self.session_summaries.get(session_id),
                )
                if summarized.summary:
                    self.session_summaries[session_id] = summarized.summary
                history = summarized.messages()
                summary_info = {
                    "summary": summarized.summary,
                    "n_compressed_turns": summarized.n_compressed,
                    "n_raw_kept": len(summarized.raw_turns) // 2,
                }
            else:
                history = raw_history

        # ===== Build user prompt (with entity context if any) =====
        evidence_for_prompt = [(c["chunk_id"], c.get("_text_full") or c["text_preview"]) for c in evidence_dicts]
        user_prompt = build_user_prompt(question, evidence_for_prompt)

        # Inject entity context as a small prefix in the system prompt
        sys_prompt = SYSTEM_PROMPT
        if self.enable_entity_tracker and session_id:
            ctx_lines = self.entities.get(session_id).context_lines()
            if ctx_lines:
                sys_prompt = SYSTEM_PROMPT + "\n\nTracked context for this user:\n" + "\n".join(ctx_lines)

        # ===== Layer: LLM generation + safety guard =====
        if self.use_llm:
            try:
                raw = call_llm(sys_prompt, user_prompt, history=history)
                decision = parse_structured(raw)
            except LLMNotConfigured as e:
                print(f"[chat] {e}; falling back to heuristic.")
                decision = heuristic_decide(question, hits)
            except Exception as e:
                print(f"[chat] LLM call failed: {e}; falling back to heuristic.")
                decision = heuristic_decide(question, hits)
        else:
            decision = heuristic_decide(question, hits)

        # Persist turn for next call
        self._persist_turn(session_id, question, decision.raw or decision.answer)

        return self._build_response(
            question=question,
            session_id=session_id,
            decision=decision.to_dict(),
            evidence=evidence_dicts,
            n_prior_turns=n_prior_turns,
            self_rag=route_decision,
            crag=crag_verdict,
            summary_info=summary_info,
            entities=entities_snapshot,
            entities_updated=entities_updated,
            skipped_retrieval=False,
            extras={
                "system_prompt": sys_prompt.strip(),
                "user_prompt_preview": user_prompt[:400] + "..." if len(user_prompt) > 400 else user_prompt,
            },
        )

    # ----- helpers -----

    def reset_session(self, session_id: str) -> None:
        self.sessions.clear(session_id)
        self.entities.clear(session_id)
        self.session_summaries.pop(session_id, None)

    def _persist_turn(self, session_id: str | None, question: str, answer: str) -> None:
        if not session_id:
            return
        buffer = self.sessions.get(session_id)
        buffer.add("user", question)
        buffer.add("assistant", answer)

    def _n_prior_turns(self, session_id: str | None) -> int:
        if not session_id:
            return 0
        return self.sessions.get(session_id).n_turns()

    def _hit_to_dict(self, h) -> dict:
        return {
            "chunk_id": h.chunk_id,
            "doc_id": h.doc_id,
            "title": h.title,
            "score": round(h.score, 4),
            "source": h.source,
            "text_preview": h.text[:240] + "..." if len(h.text) > 240 else h.text,
            "_text_full": h.text,
        }

    def _build_response(
        self,
        question: str,
        session_id: str | None,
        decision: dict,
        evidence: list,
        n_prior_turns: int,
        self_rag,
        crag,
        summary_info,
        entities: dict,
        entities_updated: list,
        skipped_retrieval: bool,
        extras: dict | None = None,
    ) -> dict:
        out = {
            "question": question,
            "session_id": session_id,
            "decision": decision,
            "evidence": [{k: v for k, v in c.items() if not k.startswith("_")} for c in evidence],
            "n_prior_turns": max(0, n_prior_turns),
            "provider": self.provider_info if self.use_llm else "heuristic-stub",
            "skipped_retrieval": skipped_retrieval,
            "self_rag": {
                "route": self_rag.route,
                "reason": self_rag.reason,
                "used_llm": self_rag.used_llm,
            },
            "crag": (
                None if crag is None
                else {
                    "quality": crag.quality,
                    "reason": crag.reason,
                    "used_llm": crag.used_llm,
                    "fallback_count": len(crag.fallback),
                }
            ),
            "summary": summary_info,
            "entities": entities,
            "entities_updated_this_turn": entities_updated,
        }
        if extras:
            out.update(extras)
        return out


def main() -> None:
    import sys, json
    bot = AmazonRAGChatbot(use_reranker=True)
    q = " ".join(sys.argv[1:]) or "How long do I have to return an item?"
    result = bot.ask(q)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
