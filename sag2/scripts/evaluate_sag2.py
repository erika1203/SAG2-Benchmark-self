#!/usr/bin/env python3
"""Evaluate saved retrieval results only; no indexing, model calls or answer generation."""

import argparse
import json
from pathlib import Path

from benchmark_data import DatasetLoader
from model_api import write_json
from sample_identity import index_unique_records, validate_question_identity

ROOT = Path(__file__).resolve().parents[1]


def evaluate(results: list[dict], records: list[dict], k_list: list[int]) -> dict:
    if not results:
        raise ValueError("No saved results to evaluate")
    if not k_list or any(k <= 0 for k in k_list):
        raise ValueError("k-list must contain positive values")
    labels = index_unique_records(records, "id", "dataset")
    index_unique_records(results, "dataset_sample_id", "saved results")
    scores = []
    for result in results:
        sample_id = result["dataset_sample_id"]
        if sample_id not in labels:
            raise ValueError(f"Result {sample_id} does not belong to the selected dataset")
        record = labels[sample_id]
        validate_question_identity(record["question"], result["question"], sample_id, "evaluation")
        gold = set(record["gold_docs"])
        if not gold:
            raise ValueError(f"Question {sample_id} has no gold evidence")
        docs = result.get("retrieved_docs")
        if not isinstance(docs, list) or any(not isinstance(d, str) for d in docs):
            raise ValueError("retrieved_docs must be a list of strings")
        docs = list(dict.fromkeys(docs))
        scores.append(
            {
                "dataset_sample_id": sample_id,
                **{f"Recall@{k}": len(gold & set(docs[:k])) / len(gold) for k in k_list},
                **{f"AllEvidence@{k}": int(gold <= set(docs[:k])) for k in k_list},
            }
        )
    return {
        "question_count": len(scores),
        "dataset_question_count": len(records),
        "scope": "saved questions only",
        "model_api_requests": 0,
        "metrics": {
            key: sum(row[key] for row in scores) / len(scores)
            for key in scores[0]
            if key != "dataset_sample_id"
        },
        "per_question": scores,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="results.json from run_sag2.py")
    parser.add_argument(
        "--dataset", choices=["musique", "2wikimultihopqa", "hotpotqa"], required=True
    )
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--k-list", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--output", type=Path, help="Default: evaluation.json next to the input")
    args = parser.parse_args()
    loader = DatasetLoader(args.dataset, str(args.dataset_dir))
    loader.validate_source_pair()
    summary = evaluate(
        json.loads(args.input.read_text(encoding="utf-8")),
        loader.get_question_records(),
        args.k_list,
    )
    output = args.output or args.input.with_name("evaluation.json")
    if output.resolve() == args.input.resolve():
        parser.error("Evaluation output must not overwrite the input results")
    write_json(output, summary)
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    print(f"评测结果：{output}")


if __name__ == "__main__":
    main()
