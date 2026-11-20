"""LangSmith evaluation datasets + experiments for retrieval, correctness, groundedness, question relevance
and personalization quality.

    export LANGSMITH_API_KEY=... LANGSMITH_TRACING=true LANGSMITH_PROJECT=rag-interview-prep
    python scripts/langsmith_eval.py --user-id <your user id>      # runs all five experiments
    python scripts/langsmith_eval.py --datasets-only               # just (re)create the datasets

Evaluators are cheap heuristics (keyword coverage / overlap) so experiments need no extra LLM spend; swap in
LLM-as-judge evaluators from `langsmith.evaluation` for deeper checks.
"""
import argparse
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from langsmith import Client, evaluate  # noqa: E402

from app.services.container import get_services  # noqa: E402

DATASETS = {
    "interview-prep/retrieval-relevance": [
        {"inputs": {"query": "What is database normalization?"}, "outputs": {"keywords": ["normal form", "redundancy"]}},
        {"inputs": {"query": "Explain deadlock conditions"}, "outputs": {"keywords": ["mutual exclusion", "circular wait"]}},
    ],
    "interview-prep/answer-correctness": [
        {"inputs": {"question": "What is a stack?", "answer": "A LIFO data structure with push and pop."},
         "outputs": {"min_score": 6}},
        {"inputs": {"question": "What is a stack?", "answer": "A kind of sorting algorithm."}, "outputs": {"max_score": 4}},
    ],
    "interview-prep/groundedness": [
        {"inputs": {"question": "What does normalization reduce?"}, "outputs": {"keywords": ["redundancy"]}},
    ],
    "interview-prep/question-relevance": [
        {"inputs": {"topic": "DBMS", "difficulty": "medium"}, "outputs": {"keywords": ["database", "sql", "transaction", "index", "normal"]}},
        {"inputs": {"topic": "Operating Systems", "difficulty": "easy"}, "outputs": {"keywords": ["process", "thread", "memory", "scheduling", "deadlock"]}},
    ],
    "interview-prep/personalization-quality": [
        {"inputs": {"scores": [9, 9, 8.5], "difficulty": "medium"}, "outputs": {"expected": "hard"}},
        {"inputs": {"scores": [3, 4, 2], "difficulty": "medium"}, "outputs": {"expected": "easy"}},
    ],
}


def ensure_datasets(client: Client) -> None:
    for name, examples in DATASETS.items():
        if client.has_dataset(dataset_name=name):
            continue
        ds = client.create_dataset(name)
        client.create_examples(inputs=[e["inputs"] for e in examples], outputs=[e["outputs"] for e in examples],
                               dataset_id=ds.id)
        print("created dataset", name)


def _cov(text: str, kws: list[str]) -> float:
    t = text.lower()
    return sum(k.lower() in t for k in kws) / max(len(kws), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", default="")
    ap.add_argument("--datasets-only", action="store_true")
    args = ap.parse_args()
    client = Client()
    ensure_datasets(client)
    if args.datasets_only:
        return
    if not args.user_id:
        sys.exit("--user-id is required (documents are per-user)")
    svc, uid = get_services(), args.user_id
    run = asyncio.run

    def t_retrieval(i):
        r = run(svc.retriever.retrieve(uid, i["query"]))
        return {"context": r.context, "n": len(r.chunks)}

    def t_correctness(i):
        return run(svc.evaluation.evaluate(question=i["question"], answer=i["answer"], topic="general", difficulty="easy"))

    def t_grounded(i):
        return run(svc.rag.answer(uid, i["question"]))

    def t_question(i):
        from app.services import prompts
        msgs = prompts.question_messages(role="", topic=i["topic"], difficulty=i["difficulty"], qtype="technical",
                                         context="", project_hint="", avoid=[], weak_note="", extra_explain=False,
                                         focus_note="", jd_note="")
        return {"question": run(svc.llm.ainvoke_text(msgs))}

    def t_personal(i):
        return {"difficulty": svc.personalization.adjust_difficulty(i["difficulty"], i["scores"])[0]}

    jobs = [
        ("interview-prep/retrieval-relevance", t_retrieval,
         lambda r, e: {"key": "retrieval_relevance", "score": _cov(r.outputs["context"], e.outputs["keywords"])}),
        ("interview-prep/answer-correctness", t_correctness,
         lambda r, e: {"key": "score_in_expected_range", "score": float(
             r.outputs["score"] >= e.outputs.get("min_score", 0) and r.outputs["score"] <= e.outputs.get("max_score", 10))}),
        ("interview-prep/groundedness", t_grounded,
         lambda r, e: {"key": "groundedness", "score": float(r.outputs["grounded"]) * (0.5 + 0.5 * _cov(r.outputs["answer"], e.outputs["keywords"]))}),
        ("interview-prep/question-relevance", t_question,
         lambda r, e: {"key": "question_relevance", "score": min(1.0, 2 * _cov(r.outputs["question"], e.outputs["keywords"]))}),
        ("interview-prep/personalization-quality", t_personal,
         lambda r, e: {"key": "difficulty_adaptation", "score": float(r.outputs["difficulty"] == e.outputs["expected"])}),
    ]
    for name, target, evaluator in jobs:
        print("running", name)
        evaluate(target, data=name, evaluators=[evaluator], experiment_prefix=name.split("/")[1])


if __name__ == "__main__":
    main()
