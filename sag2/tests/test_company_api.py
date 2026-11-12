import asyncio
from collections import Counter
from contextlib import asynccontextmanager
import json
import math

import aiohttp
from aiohttp import web
import pytest

from answer_generation import generate_answer, response_text
from api_reranker import APIReranker, sigmoid
from evaluate_sag2 import evaluate
from model_api import RequestJournal, api_endpoint
from scripts import run_sag2 as runner


@asynccontextmanager
async def serve(routes):
    app = web.Application()
    for path, handler in routes.items():
        app.router.add_post(path, handler)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await server.cleanup()


def configure(monkeypatch, base):
    settings = {
        "RERANK_BASE_URL": base + "/v1",
        "RERANK_ENDPOINT": "/rerank",
        "RERANK_MODEL": "company-reranker",
        "RERANK_API_KEY": "secret-reranker",
        "RERANK_BATCH_SIZE": "2",
        "JEV_BASE_URL": base + "/decisions",
        "JEV_MODEL": "test-jev",
        "JEV_API_KEY": "secret-jev",
        "LLM_BASE_URL": base + "/v1",
        "LLM_MODEL": "agent-model",
        "LLM_API": "chat",
        "LLM_API_KEY": "secret-agent",
        "LLM_REASONING_EFFORT": "default",
        "LLM_JSON_MODE": "true",
        "LLM_STRICT_TOOLS": "false",
        "LLM_PARALLEL_TOOL_CALLS_PARAMETER": "false",
        "ANSWER_BASE_URL": base + "/v1",
        "ANSWER_MODEL": "answer-model",
        "ANSWER_API": "chat",
        "ANSWER_API_KEY": "secret-answer",
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def arguments(tmp_path, *extra):
    return runner.build_parser().parse_args(
        [
            "--corpus",
            str(tmp_path / "corpus.json"),
            "--query",
            "Anchor?",
            "--output-dir",
            str(tmp_path / "output"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--env-file",
            str(tmp_path / "absent.env"),
            "--index-workers",
            "1",
            "--candidate-k",
            "6",
            "--seed-k",
            "3",
            "--top-k",
            "3",
            "--return-k",
            "6",
            "--expansions",
            "1",
            *extra,
        ]
    )


@pytest.mark.parametrize("scorer", ["reranker", "jev"])
@pytest.mark.parametrize("agentic", [False, True])
@pytest.mark.parametrize("task", ["retrieve", "qa"])
def test_all_switch_combinations_http_context_resume_and_no_evaluation(
    tmp_path, monkeypatch, scorer, agentic, task
):
    """Real SQLite retrieval + HTTP protocols, without any external model service."""

    async def scenario():
        calls = Counter()
        states = []
        corpus = [
            {"title": f"Passage {i}", "text": f"Anchor bridge evidence {i}."} for i in range(6)
        ]
        (tmp_path / "corpus.json").write_text(json.dumps(corpus), encoding="utf-8")

        async def rerank(request):
            calls["reranker"] += 1
            p = await request.json()
            assert request.headers["Authorization"] == "Bearer secret-reranker"
            assert p["model"] == "company-reranker"
            state = json.loads(p["query"][p["query"].index('{"original_question"') :])
            assert state["original_question"] == "Anchor?"
            states.append(state)
            assert p["top_n"] == len(p["documents"]) <= 2
            return web.json_response(
                {
                    "results": [
                        {"index": i, "relevance_score": [-2.0, 0.0][i]}
                        for i in reversed(range(len(p["documents"])))
                    ]
                }
            )

        async def jev(request):
            calls["jev"] += 1
            p = await request.json()
            assert request.headers["Authorization"] == "Bearer secret-jev"
            assert json.loads(p["state"])["original_question"] == "Anchor?"
            return web.json_response(
                {
                    "model": "test-jev",
                    "answers": {key: {"type": "noul", "noul": 0.7} for key in p["questions"]},
                }
            )

        async def llm(request):
            p = await request.json()
            if p["model"] == "answer-model":
                calls["answer"] += 1
                assert request.headers["Authorization"] == "Bearer secret-answer"
                assert "tools" not in p and "reasoning_effort" not in p
                state = json.loads(p["messages"][1]["content"])
                assert state["question"] == "Anchor?" and len(state["sources"]) == 3
                assert state["sources"][0]["number"] == 1
                message = {"role": "assistant", "content": "Evidence-based answer [1]."}
                finish = "stop"
            else:
                calls["agent"] += 1
                assert p["model"] == "agent-model"
                assert request.headers["Authorization"] == "Bearer secret-agent"
                assert "reasoning_effort" not in p and "parallel_tool_calls" not in p
                assert "strict" not in p["tools"][0]["function"]
                outputs = [json.loads(m["content"]) for m in p["messages"] if m["role"] == "tool"]
                if len(outputs) < 2:
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"search-{len(outputs)}",
                                "type": "function",
                                "function": {
                                    "name": "sag2_search",
                                    "arguments": json.dumps(
                                        {
                                            "query": "Anchor?" if not outputs else "Bridge",
                                            "confirmed_relations": [],
                                            "unresolved_relations": [],
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                    finish = "tool_calls"
                else:
                    ids = list(dict.fromkeys(p["id"] for out in outputs for p in out["passages"]))[
                        :3
                    ]
                    message = {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "ranked_ids": ids,
                                "confirmed_relations": [],
                                "unresolved_relations": [],
                            }
                        ),
                    }
                    finish = "stop"
            return web.json_response(
                {
                    "model": p["model"],
                    "choices": [{"message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 5},
                }
            )

        async with serve(
            {"/v1/rerank": rerank, "/decisions": jev, "/v1/chat/completions": llm}
        ) as base:
            configure(monkeypatch, base)
            # Disabled services must require neither credentials nor model endpoints.
            monkeypatch.delenv("JEV_API_KEY" if scorer == "reranker" else "RERANK_API_KEY")
            if not agentic:
                monkeypatch.delenv("LLM_API_KEY")
            if task == "retrieve":
                monkeypatch.delenv("ANSWER_API_KEY")
            args = arguments(
                tmp_path,
                "--scorer",
                scorer,
                "--task",
                task,
                "--agentic" if agentic else "--no-agentic",
            )
            output = await runner.run(args)
            results = json.loads((output / "custom/results.json").read_text())
            row = results[0]
            assert len(row["passages"]) == 3 and "metrics" not in row
            assert not list(output.rglob("*evaluation*")) and not list(output.rglob("metrics.json"))
            assert calls[scorer] > 0
            assert calls["jev" if scorer == "reranker" else "reranker"] == 0
            assert calls["agent"] == (3 if agentic else 0)
            assert calls["answer"] == (1 if task == "qa" else 0)
            assert ("answer" in row) == (task == "qa")
            for p in row["passages"]:
                if scorer == "reranker":
                    assert p["score"] == pytest.approx(sigmoid(p["raw_logit"]))
                else:
                    assert p["score"] == 0.7 and "raw_logit" not in p
            if agentic and scorer == "reranker":
                assert any(
                    s["current_search_query"] == "Bridge"
                    and s["retrieval_history"]
                    and s["previously_retrieved_evidence"]
                    for s in states
                )
            assert all(
                "secret-" not in f.read_text(encoding="utf-8") for f in output.rglob("*.json")
            )
            count = calls.copy()
            args.resume, args.output_dir = output, None
            await runner.run(args)
            assert calls == count
            # Simulate interruption before the per-question checkpoint, after API responses.
            (output / "custom/details.jsonl").unlink()
            await runner.run(args)
            assert calls == count
            row = json.loads((output / "custom/results.json").read_text())[0]
            assert row["api_usage"][scorer]["reused_requests"] > 0
            args.expansions = 0
            with pytest.raises(ValueError, match="Resume data"):
                await runner.run(args)
            assert calls == count

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "logit,expected", [(-1000, 0), (1000, 1), (0, 0.5), (-1, 0.26894142137), (0.5, 0.6224593312)]
)
def test_sigmoid_raw_logits_including_probability_range(logit, expected):
    assert sigmoid(logit) == pytest.approx(expected)


@pytest.mark.parametrize("logit", [True, None, "0.5", math.nan, math.inf, -math.inf])
def test_sigmoid_rejects_nonfinite_and_non_numeric_values(logit):
    with pytest.raises(ValueError, match="finite number"):
        sigmoid(logit)


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"index": 0, "relevance_score": 1}],
        [{"index": 0, "relevance_score": 1}, {"index": 0, "relevance_score": 2}],
        [{"index": -1, "relevance_score": 1}, {"index": 1, "relevance_score": 2}],
        [{"index": False, "relevance_score": 1}, {"index": 1, "relevance_score": 2}],
        [{"index": 0, "relevance_score": "1"}, {"index": 1, "relevance_score": 2}],
    ],
)
def test_reranker_invalid_indices_or_scores_fail_without_silent_drop(tmp_path, rows):
    async def handler(request):
        return web.json_response({"results": rows})

    async def scenario():
        async with serve({"/rerank": handler}) as base, aiohttp.ClientSession() as session:
            scorer = APIReranker(
                RequestJournal(session, tmp_path), base + "/rerank", "model", "secret"
            )
            with pytest.raises(ValueError):
                await scorer.score("query", [{"title": "a", "content": "b"}] * 2, [])

    asyncio.run(scenario())


def test_answer_empty_evidence_context_limit_and_responses(tmp_path):
    async def scenario():
        # A journal without a session makes any accidental network request fail.
        journal = RequestJournal(None, tmp_path)
        config = {"top_k": 3, "max_context_chars": 10}
        result = await generate_answer(journal, "问题", [], config, "unused")
        assert result["answer_status"] == "no_evidence" and result["answer_usage"]["requests"] == 0
        with pytest.raises(ValueError, match="context exceeds"):
            await generate_answer(
                journal,
                "问题",
                [{"id": 1, "source_id": "source", "title": "中文", "content": "证据"}],
                config,
                "unused",
            )

    asyncio.run(scenario())
    assert (
        response_text(
            {
                "status": "completed",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "答案 [1]"}]}
                ],
            },
            "responses",
        )
        == "答案 [1]"
    )
    for response in [
        {"choices": []},
        {"choices": [{"finish_reason": "length"}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]},
    ]:
        with pytest.raises(ValueError):
            response_text(response, "chat")


def test_index_only_needs_no_credentials_and_benchmark_evaluation_is_separate(
    tmp_path, monkeypatch
):
    async def forbidden(*args, **kwargs):
        raise AssertionError("No HTTP calls during indexing or evaluation")

    monkeypatch.setattr(aiohttp.ClientSession, "_request", forbidden)
    corpus = [{"title": "证据", "text": "Anchor bridge fact."}]
    (tmp_path / "hotpotqa_corpus.json").write_text(
        json.dumps(corpus, ensure_ascii=False), encoding="utf-8"
    )
    (tmp_path / "hotpotqa.json").write_text(
        json.dumps(
            [
                {
                    "_id": "q1",
                    "question": "Anchor?",
                    "supporting_facts": [["证据", 0]],
                    "context": [["证据", ["Anchor bridge fact."]]],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    args = runner.build_parser().parse_args(
        [
            "--dataset",
            "hotpotqa",
            "--dataset-dir",
            str(tmp_path),
            "--index-only",
            "--env-file",
            str(tmp_path / "absent.env"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "output"),
            "--index-workers",
            "1",
        ]
    )
    output = asyncio.run(runner.run(args))
    assert (output / "hotpotqa/indexed.json").exists()
    assert not (output / "hotpotqa/evaluation.json").exists()
    from benchmark_data import DatasetLoader

    records = DatasetLoader("hotpotqa", tmp_path).get_question_records()
    results = [
        {
            "dataset_sample_id": "q1",
            "question": "Anchor?",
            "retrieved_docs": ["证据\nAnchor bridge fact."],
        }
    ]
    summary = evaluate(results, records, [1, 5])
    assert summary["metrics"]["Recall@1"] == 1 and summary["model_api_requests"] == 0
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(results * 2, records, [1])
    results[0]["question"] = "Different?"
    with pytest.raises(ValueError, match="does not match"):
        evaluate(results, records, [1])


def test_answer_falls_back_to_llm_config_and_endpoint_does_not_duplicate(tmp_path, monkeypatch):
    configure(monkeypatch, "http://localhost:9999")
    for field in ("BASE_URL", "MODEL", "API", "API_KEY"):
        monkeypatch.setenv("ANSWER_" + field, "")
    args = arguments(tmp_path, "--task", "qa")
    config, secrets = runner.resolve_settings(args)
    assert "llm" not in config  # no Agentic, but its model can still generate answers
    assert config["answer"]["model"] == "agent-model"
    assert secrets["answer"] == "secret-agent"
    assert (
        api_endpoint("http://host/v1/chat/completions/", "/chat/completions")
        == "http://host/v1/chat/completions"
    )
    assert api_endpoint("http://host/v1", "/rerank") == "http://host/v1/rerank"
    args.rerank_batch_size = 0
    with pytest.raises(ValueError, match="batch size"):
        runner.resolve_settings(args)


@pytest.mark.parametrize("has_evidence", [True, False])
def test_agentic_small_corpus_and_empty_search_do_not_require_fabricated_ids(
    tmp_path, monkeypatch, has_evidence
):
    async def scenario():
        calls = Counter()
        question = "Anchor?" if has_evidence else "Xyzunfindable?"
        (tmp_path / "corpus.json").write_text(
            json.dumps([{"title": "Only", "text": "Anchor fact."}])
        )

        async def rerank(request):
            calls["reranker"] += 1
            return web.json_response({"results": [{"index": 0, "relevance_score": 2.0}]})

        async def llm(request):
            p = await request.json()
            if p["model"] == "answer-model":
                calls["answer"] += 1
                message = {"content": "Anchor fact [1]."}
                finish = "stop"
            else:
                assert "response_format" not in p
                outputs = [json.loads(m["content"]) for m in p["messages"] if m["role"] == "tool"]
                if not outputs:
                    message = {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "one",
                                "type": "function",
                                "function": {
                                    "name": "sag2_search",
                                    "arguments": json.dumps(
                                        {
                                            "query": question,
                                            "confirmed_relations": [],
                                            "unresolved_relations": [],
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                    finish = "tool_calls"
                else:
                    message = {
                        "content": json.dumps(
                            {
                                "ranked_ids": [p["id"] for p in outputs[0]["passages"]],
                                "confirmed_relations": [],
                                "unresolved_relations": [],
                            }
                        )
                    }
                    finish = "stop"
            return web.json_response({"choices": [{"message": message, "finish_reason": finish}]})

        async with serve({"/v1/rerank": rerank, "/v1/chat/completions": llm}) as base:
            configure(monkeypatch, base)
            args = arguments(
                tmp_path, "--task", "qa", "--agentic", "--no-llm-json-mode", "--expansions", "0"
            )
            args.query = question
            output = await runner.run(args)
            row = json.loads((output / "custom/results.json").read_text())[0]
            assert len(row["passages"]) == int(has_evidence)
            assert calls["answer"] == int(has_evidence)
            assert row["answer_status"] == ("generated" if has_evidence else "no_evidence")

    asyncio.run(scenario())


def test_benchmark_inputs_only_produce_results_until_evaluation_is_requested(tmp_path, monkeypatch):
    async def scenario():
        question = "Anchor?"
        (tmp_path / "hotpotqa_corpus.json").write_text(
            json.dumps([{"title": "One", "text": "Anchor fact."}])
        )
        (tmp_path / "hotpotqa.json").write_text(
            json.dumps(
                [
                    {
                        "_id": "original-id",
                        "question": question,
                    }
                ]
            )
        )
        calls = []

        async def rerank(request):
            p = await request.json()
            assert "supporting_facts" not in json.dumps(p) and "gold_docs" not in json.dumps(p)
            calls.append(p)
            return web.json_response({"results": [{"index": 0, "relevance_score": -1}]})

        async with serve({"/v1/rerank": rerank}) as base:
            configure(monkeypatch, base)
            args = runner.build_parser().parse_args(
                [
                    "--dataset",
                    "hotpotqa",
                    "--dataset-dir",
                    str(tmp_path),
                    "--output-dir",
                    str(tmp_path / "output"),
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--env-file",
                    str(tmp_path / "absent.env"),
                    "--index-workers",
                    "1",
                    "--expansions",
                    "0",
                ]
            )
            output = await runner.run(args)
            results = json.loads((output / "hotpotqa/results.json").read_text())
            assert "metrics" not in results[0] and "answer" not in results[0]
            assert not list(output.rglob("evaluation.json"))
            # Labels are supplied only to the later, independent evaluation step.
            summary = evaluate(
                results,
                [{"id": "original-id", "question": question, "gold_docs": ["One\nAnchor fact."]}],
                [1],
            )
            assert summary["metrics"]["Recall@1"] == 1
            assert len(calls) == 1

    asyncio.run(scenario())
