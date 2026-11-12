"""Remote /rerank adapter: raw logits -> sigmoid -> SAG2 evidence scores."""

import math

from model_api import RequestJournal, token_usage
from sag2 import CRITERION, EVIDENCE_SCORE_NOTE, compact_json


def sigmoid(logit: float) -> float:
    if type(logit) not in (int, float) or not math.isfinite(logit):
        raise ValueError("Reranker logit must be a finite number")
    if logit >= 0:
        return 1 / (1 + math.exp(-logit))
    exp_value = math.exp(logit)
    return exp_value / (1 + exp_value)


class APIReranker:
    service = "reranker"

    def __init__(
        self, journal: RequestJournal, endpoint: str, model: str, api_key: str, batch_size: int = 32
    ):
        if batch_size < 1:
            raise ValueError("Reranker batch size must be positive")
        self.journal, self.endpoint, self.model, self.api_key = journal, endpoint, model, api_key
        self.batch_size = batch_size

    async def score(
        self, query: str, candidates: list[dict], evidence: list[dict], *, context=None
    ):
        # Preserve the original question and previous evidence during both expansion
        # and Agentic query rewriting, as the Jev scorer does.
        state = {
            "original_question": context["original_question"] if context is not None else query,
            "previously_retrieved_evidence": [
                {
                    "title": p["title"],
                    "passage": p["content"],
                    **(
                        {"previous_score": p["previous_jev_score"]}
                        if "previous_jev_score" in p
                        else {}
                    ),
                }
                for p in evidence
            ],
        }
        if any("previous_jev_score" in p for p in evidence):
            state["evidence_score_note"] = EVIDENCE_SCORE_NOTE
        if context is not None:
            state.update(
                current_search_query=query,
                retrieval_history=context["search_history"],
                previous_decisions=context["previous_decisions"],
            )
            combined = {p["id"]: p for p in [*context["evidence"], *evidence]}
            state["previously_retrieved_evidence"] = [
                {
                    "passage_id": p["id"],
                    "title": p["title"],
                    "passage": p["content"],
                    **(
                        {"previous_score": p["previous_jev_score"]}
                        if "previous_jev_score" in p
                        else {}
                    ),
                }
                for p in combined.values()
            ]
        full_query = CRITERION + "\n\n" + compact_json(state)
        scored, batches = [], []
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            payload = {
                "model": self.model,
                "query": full_query,
                "documents": [p["title"] + "\n" + p["content"] for p in batch],
                # SAG2 needs a score for every candidate, including low-scoring ones.
                "top_n": len(batch),
            }
            response = await self.journal.post(self.service, self.endpoint, self.api_key, payload)
            rows = response.get("results")
            if not isinstance(rows, list) or len(rows) != len(batch):
                raise ValueError("Reranker must return a result for every input document")
            logits = {}
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("Each reranker result must be an object")
                i = row.get("index")
                if type(i) is not int or not 0 <= i < len(batch) or i in logits:
                    raise ValueError("Reranker returned duplicate or invalid document indices")
                logit = row.get("relevance_score", row.get("score"))
                logits[i] = (logit, sigmoid(logit))
            model = response.get("model") or self.model
            scored.extend(
                dict(p, rerank_logit=logits[i][0], jev_score=logits[i][1], jev_model=model)
                for i, p in enumerate(batch)
            )
            batches.append(
                {
                    "model": model,
                    "candidate_count": len(batch),
                    "reused": self.journal.records[-1]["reused"],
                    "score_transform": "sigmoid",
                    **token_usage(response),
                }
            )
        return scored, {
            "requests": len(batches),
            "judgments": len(candidates),
            "batches": batches,
            "input_tokens": sum(b["input_tokens"] for b in batches),
            "output_tokens": sum(b["output_tokens"] for b in batches),
        }
