#!/usr/bin/env python3
"""Run retrieval or retrieval + answer generation. Evaluation is a separate command."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

from agentic_sag2 import run_agentic
from answer_generation import generate_answer
from api_reranker import APIReranker
from benchmark_data import DatasetLoader
from model_api import RecordedJev, RequestJournal, api_endpoint, write_json
from sag2 import CorpusIndex, DEFAULT_CHUNK_CHARS, compact_json, corpus_fingerprint, search_sag2
from sample_identity import index_unique_records, validate_question_identity

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("musique", "2wikimultihopqa", "hotpotqa")
DATA_RELEASE = "5ec05b38deecc3318bb432c69865959c56058990"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus", type=Path, help='JSON list of {"title": ..., "text": ...}')
    source.add_argument(
        "--dataset", choices=[*DATASETS, "all"], help="Use benchmark inputs without evaluation"
    )
    questions = parser.add_mutually_exclusive_group()
    questions.add_argument("--query", help="One question for a custom corpus")
    questions.add_argument(
        "--questions", type=Path, help='JSON list of {"id": ..., "question": ...}'
    )
    parser.add_argument("--task", choices=["retrieve", "qa"], default="retrieve")
    parser.add_argument("--scorer", choices=["reranker", "jev"], default="reranker")
    parser.add_argument("--agentic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--env-file", type=Path, default=Path(os.getenv("SAG_ENV_FILE", ROOT / ".env"))
    )
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument(
        "--download", action="store_true", help="Download missing fixed benchmark inputs"
    )
    parser.add_argument(
        "--index-only", action="store_true", help="Build indices without any model calls"
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "sag2")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume", type=Path, help="Resume a run with the same data and model settings"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--index-workers", type=int, default=2)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--seed-k", type=int, default=10)
    parser.add_argument("--expansions", type=int, choices=[0, 1, 2])
    parser.add_argument("--expansion-term-limit", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--max-tool-calls", type=int, default=3)
    parser.add_argument("--return-k", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument(
        "--trust-env", action="store_true", help="Honor HTTP(S)_PROXY/NO_PROXY in aiohttp"
    )
    parser.add_argument("--rerank-base-url")
    parser.add_argument(
        "--rerank-endpoint", help="Path relative to base URL, or a full endpoint URL"
    )
    parser.add_argument("--rerank-model")
    parser.add_argument("--rerank-batch-size", type=int)
    parser.add_argument("--jev-base-url")
    parser.add_argument("--jev-model")
    parser.add_argument("--llm-base-url", help="Agentic LLM API base, normally including /v1")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-api", choices=["chat", "responses"])
    parser.add_argument(
        "--reasoning-effort", choices=["default", "none", "low", "medium", "high", "xhigh", "max"]
    )
    parser.add_argument("--llm-json-mode", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--answer-base-url")
    parser.add_argument("--answer-model")
    parser.add_argument("--answer-api", choices=["chat", "responses"])
    parser.add_argument("--answer-top-k", type=int, default=5)
    parser.add_argument("--answer-max-context-chars", type=int, default=60000)
    return parser


def validate_args(args) -> None:
    for field in (
        "concurrency",
        "index_workers",
        "candidate_k",
        "seed_k",
        "expansion_term_limit",
        "top_k",
        "chunk_chars",
        "max_tool_calls",
        "return_k",
        "timeout",
        "answer_top_k",
        "answer_max_context_chars",
    ):
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.chunk_chars < 2:
        raise ValueError("--chunk-chars must be at least 2")
    if args.seed_k > args.candidate_k or args.top_k > args.candidate_k:
        raise ValueError("seed-k and top-k cannot exceed candidate-k")
    if args.agentic and not args.top_k <= args.return_k <= args.candidate_k:
        raise ValueError("Agentic mode requires top-k <= return-k <= candidate-k")
    if args.offset < 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("offset must be nonnegative and limit must be positive")
    if args.resume and args.output_dir:
        raise ValueError("Use either --resume or --output-dir")
    if args.dataset and (args.query is not None or args.questions):
        raise ValueError(
            "--query/--questions are for --corpus; --dataset supplies its own questions"
        )
    if args.corpus and not args.index_only and args.query is None and args.questions is None:
        raise ValueError("--corpus requires --query or --questions")
    if args.query is not None and not args.query.strip():
        raise ValueError("--query must not be blank")
    if args.download and not args.dataset:
        raise ValueError("--download requires --dataset")


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).lower()
    if value not in ("true", "false", "1", "0"):
        raise ValueError(f"{name} must be true or false")
    return value in ("true", "1")


def resolve_settings(args) -> tuple[dict, dict]:
    """Keep keys out of the persisted settings and request payloads."""
    load_dotenv(args.env_file)
    settings = {
        "task": args.task,
        "scorer": args.scorer,
        "agentic": args.agentic,
        "retrieval": {
            "candidate_k": args.candidate_k,
            "seed_k": args.seed_k,
            "max_hops": args.expansions
            if args.expansions is not None
            else (1 if args.agentic else 2),
            "expansion_term_limit": args.expansion_term_limit,
            "top_k": args.top_k,
            "return_k": args.return_k,
            "max_tool_calls": args.max_tool_calls,
            "allow_fewer_results": True,
        },
        "chunk_chars": args.chunk_chars,
    }
    secrets = {}
    if args.index_only:
        return settings, secrets
    if args.scorer == "reranker":
        base = args.rerank_base_url or os.getenv("RERANK_BASE_URL", "")
        route = args.rerank_endpoint or os.getenv("RERANK_ENDPOINT", "/rerank")
        if not base and not route.startswith(("http://", "https://")):
            raise ValueError("Set RERANK_BASE_URL or a full RERANK_ENDPOINT URL")
        endpoint = (
            api_endpoint(route, "")
            if route.startswith(("http://", "https://"))
            else api_endpoint(base, "/" + route.lstrip("/"))
        )
        model = args.rerank_model or os.getenv("RERANK_MODEL") or os.getenv("RERANK_MODEL_NAME", "")
        batch_size = (
            args.rerank_batch_size
            if args.rerank_batch_size is not None
            else int(os.getenv("RERANK_BATCH_SIZE", "32"))
        )
        if batch_size <= 0:
            raise ValueError("Reranker batch size must be positive")
        settings["reranker"] = {
            "endpoint": endpoint,
            "model": model,
            "batch_size": batch_size,
            "score_transform": "sigmoid",
        }
        secrets["reranker"] = os.getenv("RERANK_API_KEY", "")
    else:
        settings["jev"] = {
            "endpoint": args.jev_base_url
            or os.getenv("JEV_BASE_URL", "https://openrouter.ai/api/alpha/decisions"),
            "model": args.jev_model or os.getenv("JEV_MODEL", "typesafe/jev-1.13"),
        }
        api_endpoint(settings["jev"]["endpoint"], "")
        secrets["jev"] = os.getenv("JEV_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
    llm = {
        "base_url": args.llm_base_url or os.getenv("LLM_BASE_URL", ""),
        "model": args.llm_model or os.getenv("LLM_MODEL", ""),
        "api": args.llm_api or os.getenv("LLM_API", "chat"),
    }
    llm_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    if args.agentic:
        settings["llm"] = {
            **llm,
            "reasoning_effort": args.reasoning_effort
            or os.getenv("LLM_REASONING_EFFORT", "default"),
            "strict_tools": env_bool("LLM_STRICT_TOOLS", False),
            "parallel_tool_calls_parameter": env_bool("LLM_PARALLEL_TOOL_CALLS_PARAMETER", False),
            "json_mode": args.llm_json_mode
            if args.llm_json_mode is not None
            else env_bool("LLM_JSON_MODE", True),
        }
        secrets["llm"] = llm_key
        if settings["llm"]["reasoning_effort"] not in (
            "default",
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ):
            raise ValueError("Invalid LLM_REASONING_EFFORT")
    if args.task == "qa":
        settings["answer"] = {
            "base_url": args.answer_base_url or os.getenv("ANSWER_BASE_URL") or llm["base_url"],
            "model": args.answer_model or os.getenv("ANSWER_MODEL") or llm["model"],
            "api": args.answer_api or os.getenv("ANSWER_API") or llm["api"],
            "top_k": min(args.answer_top_k, args.top_k),
            "max_context_chars": args.answer_max_context_chars,
        }
        secrets["answer"] = os.getenv("ANSWER_API_KEY") or llm_key
    for service, key in secrets.items():
        if not key:
            raise ValueError(
                f"Missing API key for {service}; configure its *_API_KEY in --env-file"
            )
        if not settings[service]["model"].strip():
            raise ValueError(f"Missing model ID for {service}")
        if service in ("llm", "answer"):
            config = settings[service]
            if config["api"] not in ("chat", "responses"):
                raise ValueError(f"{service} API must be chat or responses")
            api_endpoint(
                config["base_url"], "/chat/completions" if config["api"] == "chat" else "/responses"
            )
    return settings, secrets


async def download_inputs(args) -> None:
    names = DATASETS if args.dataset == "all" else (args.dataset,)
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession(
        trust_env=args.trust_env, timeout=aiohttp.ClientTimeout(total=600)
    ) as session:
        for name in names:
            for suffix in (".json", "_corpus.json"):
                path = args.dataset_dir / (name + suffix)
                if path.exists():
                    continue
                url = f"https://huggingface.co/datasets/osunlp/HippoRAG_2/resolve/{DATA_RELEASE}/{path.name}"
                async with session.get(url) as response:
                    response.raise_for_status()
                    with path.with_suffix(".part").open("wb") as stream:
                        async for block in response.content.iter_chunked(1024 * 1024):
                            stream.write(block)
                path.with_suffix(".part").replace(path)


def load_inputs(args) -> list[dict]:
    inputs = []
    if args.corpus:
        corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
        if (
            not isinstance(corpus, list)
            or not corpus
            or any(
                not isinstance(p, dict)
                or not isinstance(p.get("title"), str)
                or not isinstance(p.get("text"), str)
                for p in corpus
            )
        ):
            raise ValueError('Corpus must be a nonempty JSON list of {"title": ..., "text": ...}')
        records = (
            json.loads(args.questions.read_text(encoding="utf-8"))
            if args.questions
            else ([{"id": "query-1", "question": args.query}] if args.query is not None else [])
        )
        inputs.append(
            {
                "name": "custom",
                "corpus": corpus,
                "records": records,
                "corpus_sha256": corpus_fingerprint(args.corpus),
            }
        )
    else:
        for name in DATASETS if args.dataset == "all" else (args.dataset,):
            loader = DatasetLoader(name, str(args.dataset_dir))
            loader.validate_source_pair()
            # Read only questions here: inference does not require gold labels at all.
            records = [
                {"id": r.get("_id", r.get("id")), "question": r["question"]}
                for r in loader.load_samples()
            ]
            inputs.append(
                {
                    "name": name,
                    "corpus": loader.load_corpus(),
                    "records": records,
                    "corpus_sha256": corpus_fingerprint(args.dataset_dir / f"{name}_corpus.json"),
                }
            )
    for data in inputs:
        records = data["records"]
        if not isinstance(records, list) or any(
            not isinstance(r, dict)
            or not isinstance(r.get("id"), str)
            or not r["id"].strip()
            or not isinstance(r.get("question"), str)
            or not r["question"].strip()
            for r in records
        ):
            raise ValueError(
                'Questions must be a list of nonempty string "id" and "question" fields'
            )
        index_unique_records(records, "id", "input questions")
        data["records"] = [{"id": r["id"].strip(), "question": r["question"]} for r in records][
            args.offset : None if args.limit is None else args.offset + args.limit
        ]
        if not data["records"] and not args.index_only:
            raise ValueError("Requested question range is empty")
    return inputs


async def infer(index, question, settings, secrets, journal) -> dict:
    config = settings[settings["scorer"]]
    if settings["scorer"] == "reranker":
        scorer = APIReranker(
            journal, config["endpoint"], config["model"], secrets["reranker"], config["batch_size"]
        )
    else:
        scorer = RecordedJev(journal, config["endpoint"], config["model"], secrets["jev"])
    options = settings["retrieval"]
    if settings["agentic"]:
        result = await run_agentic(
            index, scorer, question, journal, settings["llm"], secrets["llm"], options
        )
    else:
        result = await search_sag2(
            index,
            scorer,
            question,
            **{
                k: options[k] for k in ("candidate_k", "seed_k", "max_hops", "expansion_term_limit")
            },
        )
    documents = {}
    for passage in result.pop("candidates"):
        documents.setdefault(passage["document_id"], passage)
    passages = list(documents.values())[: options["top_k"]]
    result.pop("baseline")
    result.update(
        question=question,
        task=settings["task"],
        scorer=settings["scorer"],
        agentic=settings["agentic"],
        retrieved_docs=[p["full_doc"] for p in passages],
        passages=[
            {
                "rank": i,
                "id": p["id"],
                "source_id": p["source_id"],
                "title": p["title"],
                "heading": p.get("heading", ""),
                "text": p["content"],
                "score": p["jev_score"],
                **({"raw_logit": p["rerank_logit"]} if "rerank_logit" in p else {}),
            }
            for i, p in enumerate(passages, 1)
        ],
    )
    if settings["task"] == "qa":
        result.update(
            await generate_answer(
                journal, question, passages, settings["answer"], secrets["answer"]
            )
        )
    result["api_usage"] = {service: journal.summary(service) for service in secrets}
    return result


async def run(args) -> Path:
    validate_args(args)
    settings, secrets = resolve_settings(args)
    if args.download:
        await download_inputs(args)
    inputs = load_inputs(args)
    manifest = {
        "settings": settings,
        "inputs": [{k: d[k] for k in ("name", "corpus_sha256", "records")} for d in inputs],
        "code_sha256": {
            name: corpus_fingerprint(ROOT / "scripts" / name)
            for name in (
                "run_sag2.py",
                "sag2.py",
                "chunking.py",
                "sag2_rank.py",
                "agentic_sag2.py",
                "api_reranker.py",
                "answer_generation.py",
                "model_api.py",
            )
        },
        "prompt_sha256": {
            p.name: corpus_fingerprint(p) for p in sorted((ROOT / "prompts").glob("agentic*"))
        },
        "tool_sha256": corpus_fingerprint(ROOT / "prompts/sag2_tool.json"),
    }
    output = (
        args.resume
        or args.output_dir
        or ROOT / "output" / "rag" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    )
    if args.resume:
        saved = json.loads((output / "config.json").read_text(encoding="utf-8"))
        if manifest != saved:
            raise ValueError(
                "Resume data, code or settings changed; use identical settings or a new output directory"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "config.json", manifest)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    async with aiohttp.ClientSession(
        connector=connector,
        trust_env=args.trust_env,
        timeout=aiohttp.ClientTimeout(total=args.timeout),
    ) as session:
        for data in inputs:
            folder = output / data["name"]
            folder.mkdir(exist_ok=True)
            index = CorpusIndex(
                args.cache_dir
                / f"{data['name']}-{data['corpus_sha256'][:16]}-{args.chunk_chars}.sqlite",
                data["corpus"],
                data["corpus_sha256"],
                args.chunk_chars,
                index_workers=args.index_workers,
            )
            try:
                if args.index_only:
                    write_json(
                        folder / "indexed.json",
                        {"documents": index.document_count, "chunks": index.total},
                    )
                    continue
                checkpoint = folder / "details.jsonl"
                completed = (
                    [
                        json.loads(line)
                        for line in checkpoint.read_text(encoding="utf-8").splitlines()
                    ]
                    if checkpoint.exists()
                    else []
                )
                by_id = index_unique_records(completed, "dataset_sample_id", "checkpoint")
                records = data["records"]
                if set(by_id) - {r["id"] for r in records}:
                    raise ValueError("Checkpoint contains unexpected question IDs")
                for record in records:
                    if record["id"] in by_id:
                        validate_question_identity(
                            record["question"],
                            by_id[record["id"]]["question"],
                            record["id"],
                            "checkpoint",
                        )
                pending = iter((i, r) for i, r in enumerate(records) if r["id"] not in by_id)
                with checkpoint.open("a", encoding="utf-8") as stream:

                    async def worker():
                        for i, record in pending:
                            journal = RequestJournal(session, folder / "requests" / str(i))
                            result = await infer(
                                index, record["question"], settings, secrets, journal
                            )
                            result["dataset_sample_id"] = record["id"]
                            stream.write(compact_json(result) + "\n")
                            stream.flush()
                            by_id[record["id"]] = result
                            print(f"{data['name']}: {len(by_id)}/{len(records)} 完成", flush=True)

                    async with asyncio.TaskGroup() as group:
                        for _ in range(args.concurrency):
                            group.create_task(worker())
                results = [by_id[r["id"]] for r in records]
                write_json(folder / "results.json", results)
                if args.query is not None and args.task == "qa":
                    print(results[0]["answer"], flush=True)
            finally:
                index.close()
    print(f"结果目录：{output}（未执行评测）", flush=True)
    return output


def main():
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        asyncio.run(run(args))
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
