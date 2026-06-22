"""Run UrdWell end to end on the cleaned LongMemEval benchmark.

The runner replays each timestamped history into an isolated UrdWell
store, extracts and consolidates structured memories, retrieves relevant
memories, asks a reader model to answer, and applies the official LongMemEval
LLM-as-a-judge rubric.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from urdwell import embeddings
from urdwell import pipeline
from urdwell import ranking
from urdwell.models import Memory, VALID_MEMORY_TYPES
from urdwell.storage import ParquetStore

from benchmarks.longmemeval.llm_client import ChatClient, ChatResponse
from benchmarks.longmemeval.run_retrieval import (
    DEFAULT_DATASET,
    question_type,
    select_entries,
)


DEFAULT_RUNS_DIR = PROJECT_ROOT / "benchmarks" / "longmemeval" / "reports" / "e2e"
PROMPT_VERSION = "urdwell-longmemeval-v1"
DATE_FORMAT = "%Y/%m/%d (%a) %H:%M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--system",
        choices=["context-memory", "raw-history", "no-memory"],
        default="context-memory",
    )
    parser.add_argument(
        "--ingestion-mode",
        choices=["llm", "verbatim"],
        default="llm",
        help="How UrdWell converts sessions into stored memories.",
    )
    parser.add_argument("--reader-model", required=True)
    parser.add_argument("--extractor-model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--base-url",
        default=os.getenv(
            "URDWELL_LLM_BASE_URL",
            os.getenv("CONTEXT_MEMORY_LLM_BASE_URL", "https://api.openai.com/v1"),
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the provider API key.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
    )
    parser.add_argument(
        "--backend",
        choices=["fastembed", "hashing"],
        default="fastembed",
    )
    parser.add_argument(
        "--ranking",
        choices=["cosine", "hybrid"],
        default="cosine",
        help="Memory retrieval ranking: dense cosine, or hybrid BM25+cosine (RRF).",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=ranking.DEFAULT_POOL_SIZE,
        help="Hybrid only: top-cosine candidates the lexical leg may reorder.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=ranking.RRF_K,
        help="Hybrid only: Reciprocal Rank Fusion constant.",
    )
    parser.add_argument(
        "--reuse-stores-from",
        default=None,
        help=(
            "Reuse per-case memory stores from an existing run name instead of "
            "re-ingesting. Skips extraction and arbitration; only retrieval, "
            "answer generation, and judging run again."
        ),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=pipeline.SIMILARITY_THRESHOLD)
    parser.add_argument(
        "--include-expired",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-large-llm-ingestion",
        action="store_true",
        help="Allow LLM extraction on datasets containing over 5,000 sessions.",
    )
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        stripped = stripped.rsplit("```", 1)[0]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end < start:
        raise ValueError(f"model did not return a JSON object: {text[:300]!r}")
    parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model response must contain a JSON object")
    return parsed


def usage_dict(response: ChatResponse) -> dict[str, int]:
    return {
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
    }


def add_usage(total: dict[str, int], response: ChatResponse) -> None:
    total["prompt_tokens"] += response.prompt_tokens
    total["completion_tokens"] += response.completion_tokens
    total["requests"] += 1


def clean_session(session: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": str(turn["role"]), "content": str(turn["content"])}
        for turn in session
    ]


def sorted_sessions(entry: dict[str, Any]) -> list[tuple[str, str, list[dict]]]:
    sessions = list(
        zip(
            entry["haystack_dates"],
            entry["haystack_session_ids"],
            entry["haystack_sessions"],
        )
    )
    return sorted(sessions, key=lambda item: datetime.strptime(item[0], DATE_FORMAT))


def extraction_prompt(
    session_date: str,
    session: list[dict[str, Any]],
) -> str:
    return f"""Extract durable memories from this dated user-assistant session.

Return one JSON object with this exact shape:
{{"memories": [{{"content": "one self-contained sentence", "type": "fact|preference|decision|temporary_state", "confidence": 0.0}}]}}

Rules:
- Capture information from both user and assistant turns that could matter in a future conversation.
- Preserve names, quantities, dates, sequences, preferences, decisions, and changes.
- Each memory must be understandable without the original conversation.
- Do not invent information.
- Return an empty list when nothing is worth remembering.

Session date: {session_date}
Session:
{json.dumps(clean_session(session), ensure_ascii=False)}
"""


def normalize_extracted_memories(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_memories = payload.get("memories", [])
    if not isinstance(raw_memories, list):
        raise ValueError("memories must be a list")

    normalized = []
    for item in raw_memories:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        memory_type = str(item.get("type", "fact"))
        if memory_type not in VALID_MEMORY_TYPES:
            memory_type = "fact"
        try:
            confidence = float(item.get("confidence", 0.8))
        except (TypeError, ValueError):
            confidence = 0.8
        normalized.append(
            {
                "content": content,
                "type": memory_type,
                "confidence": max(0.0, min(confidence, 1.0)),
            }
        )
    return normalized


def extraction_cache_key(
    model: str,
    session_date: str,
    session: list[dict[str, Any]],
) -> str:
    serialized = json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "date": session_date,
            "session": clean_session(session),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def extract_session_memories(
    client: ChatClient,
    cache_dir: Path,
    session_date: str,
    session: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    key = extraction_cache_key(client.model, session_date, session)
    cache_path = cache_dir / f"{key}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return cached["memories"], {"cache_hit": True, "usage": {}}

    prompt = extraction_prompt(session_date, session)
    response = client.complete(prompt, max_tokens=1200, json_mode=True)
    memories = normalize_extracted_memories(extract_json_object(response.content))
    atomic_write_json(
        cache_path,
        {
            "model": client.model,
            "prompt_version": PROMPT_VERSION,
            "memories": memories,
        },
    )
    return memories, {
        "cache_hit": False,
        "usage": usage_dict(response),
        "raw_response": response.content,
    }


def arbitration_prompt(
    new_memory: Memory,
    candidates: list[dict[str, Any]],
) -> str:
    return f"""Resolve a long-term memory consolidation decision.

Return JSON only:
{{"decision": "ADD|IGNORE|EXPIRE", "target_id": "candidate id or null"}}

Use ADD when the memories are compatible and both should remain.
Use IGNORE when the new memory is a duplicate.
Use EXPIRE when the new memory supersedes or corrects one candidate.
IGNORE and EXPIRE require the selected candidate ID. ADD requires null.

New memory:
{json.dumps(new_memory.to_dict(), ensure_ascii=False)}

Candidates:
{json.dumps(candidates, ensure_ascii=False)}
"""


def arbitrate_memory(
    client: ChatClient,
    memory: Memory,
    report: dict[str, Any],
) -> tuple[str, str | None, ChatResponse]:
    response = client.complete(
        arbitration_prompt(memory, report["candidates"]),
        max_tokens=120,
        json_mode=True,
    )
    payload = extract_json_object(response.content)
    decision = str(payload.get("decision", "ADD")).upper()
    target_id = payload.get("target_id")
    candidate_ids = {item["id"] for item in report["candidates"]}
    if decision not in pipeline.ARBITRATION_ACTIONS:
        decision = "ADD"
    if decision == "ADD":
        target_id = None
    elif target_id not in candidate_ids:
        decision = "ADD"
        target_id = None
    return decision, target_id, response


def ingest_case(
    entry: dict[str, Any],
    store: ParquetStore,
    extractor: ChatClient,
    cache_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    logs = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
    for session_date, session_id, session in sorted_sessions(entry):
        source = f"{session_id} @ {session_date}"
        for turn in session:
            store.append_archive(turn["role"], turn["content"], source)

        extracted, extraction_log = extract_session_memories(
            extractor,
            cache_dir,
            session_date,
            session,
        )
        if extraction_log.get("usage"):
            usage["prompt_tokens"] += extraction_log["usage"]["prompt_tokens"]
            usage["completion_tokens"] += extraction_log["usage"]["completion_tokens"]
            usage["requests"] += 1

        memory_logs = []
        for item in extracted:
            memory = Memory(
                content=item["content"],
                type=item["type"],
                source=source,
                confidence=item["confidence"],
                valid_from=session_date,
            )
            report = pipeline.process_memory(store, memory)
            arbitration = None
            if report["action"] == "ARBITRATION_REQUIRED":
                decision, target_id, response = arbitrate_memory(
                    extractor,
                    memory,
                    report,
                )
                add_usage(usage, response)
                arbitration = {
                    "decision": decision,
                    "target_id": target_id,
                    "usage": usage_dict(response),
                }
                report = pipeline.process_memory(
                    store,
                    memory,
                    decision=decision,
                    target_id=target_id,
                )
            memory_logs.append(
                {
                    "memory": item,
                    "result": report,
                    "arbitration": arbitration,
                }
            )

        logs.append(
            {
                "session_id": session_id,
                "session_date": session_date,
                "turns": len(session),
                "extraction": extraction_log,
                "memories": memory_logs,
            }
        )
    return logs, usage


def session_rounds(session: list[dict[str, Any]]) -> list[tuple[int, list[dict]]]:
    """Group each user turn with the assistant turns that follow it."""
    rounds: list[tuple[int, list[dict]]] = []
    current_index: int | None = None
    current_turns: list[dict] = []
    for index, turn in enumerate(session):
        if turn["role"] == "user":
            if current_turns and current_index is not None:
                rounds.append((current_index, current_turns))
            current_index = index
            current_turns = [turn]
        elif current_turns:
            current_turns.append(turn)
    if current_turns and current_index is not None:
        rounds.append((current_index, current_turns))
    return rounds


def ingest_verbatim_case(
    entry: dict[str, Any],
    store: ParquetStore,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Store dated conversational rounds without an extraction LLM."""
    memories: list[Memory] = []
    logs = []
    for session_date, session_id, session in sorted_sessions(entry):
        source = f"{session_id} @ {session_date}"
        for turn in session:
            store.append_archive(turn["role"], turn["content"], source)

        round_count = 0
        for turn_index, turns in session_rounds(session):
            role_lines = "\n".join(
                f"{turn['role'].capitalize()}: {turn['content'].strip()}"
                for turn in turns
            )
            memories.append(
                Memory(
                    content=f"Session date: {session_date}\n{role_lines}",
                    type="fact",
                    source=f"{session_id}_{turn_index + 1}",
                    confidence=1.0,
                    valid_from=session_date,
                )
            )
            round_count += 1
        logs.append(
            {
                "session_id": session_id,
                "session_date": session_date,
                "turns": len(session),
                "stored_rounds": round_count,
            }
        )

    vectors = embeddings.embed_many([memory.content for memory in memories])
    store.add_many(memories, vectors)
    return logs, {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}


def retrieve_memories(
    store: ParquetStore,
    query: str,
    *,
    k: int,
    threshold: float,
    include_expired: bool,
    strategy: str = "cosine",
    pool_size: int = ranking.DEFAULT_POOL_SIZE,
    rrf_k: int = ranking.RRF_K,
) -> list[dict[str, Any]]:
    query_embedding = embeddings.embed(query)
    stored_embeddings = store.all_embeddings()
    candidates = [
        (memory, stored_embeddings[memory.id])
        for memory in store.all(active_only=not include_expired)
        if memory.id in stored_embeddings
    ]

    if strategy == "hybrid":
        # Dense + lexical fusion (RRF). Abstention stays a pure cosine decision,
        # so ``threshold`` keeps the same meaning as the cosine path below.
        scored = ranking.hybrid_rank(
            query,
            query_embedding,
            candidates,
            k,
            cosine_floor=threshold,
            pool_size=pool_size,
            rrf_k=rrf_k,
        )
    else:
        ranked = sorted(
            (
                (memory, embeddings.cosine_similarity(query_embedding, vector))
                for memory, vector in candidates
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        scored = [pair for pair in ranked[:k] if pair[1] >= threshold]

    return [
        {**memory.to_dict(), "score": round(score, 4)}
        for memory, score in scored
    ]


def answer_prompt(
    entry: dict[str, Any],
    memories: list[dict[str, Any]],
) -> str:
    return f"""Answer the question using only the retrieved long-term memories.

First identify the relevant memories and reason over dates, updates, and
multi-session evidence. Give a concise final answer. If the memories do not
contain enough information, explicitly say that the answer cannot be
determined. Do not invent missing facts.

Retrieved memories:
{json.dumps(memories, ensure_ascii=False, indent=2)}

Current date: {entry["question_date"]}
Question: {entry["question"]}
Answer:
"""


def raw_history_prompt(entry: dict[str, Any]) -> str:
    history = [
        {
            "session_date": date,
            "session": clean_session(session),
        }
        for date, _, session in sorted_sessions(entry)
    ]
    return f"""I will give you history chats between you and a user.
Answer the question based on the relevant chat history. First extract the
relevant information, then reason over dates and updates. If the history does
not contain enough information, explicitly say that the answer cannot be
determined.

History chats:
{json.dumps(history, ensure_ascii=False)}

Current date: {entry["question_date"]}
Question: {entry["question"]}
Answer:
"""


def no_memory_prompt(entry: dict[str, Any]) -> str:
    return f"""Answer this question about a user's prior conversations.
If no supplied information supports an answer, explicitly say that the answer
cannot be determined. Do not invent personal information.

Current date: {entry["question_date"]}
Question: {entry["question"]}
Answer:
"""


def answer_without_context_memory(
    entry: dict[str, Any],
    reader: ChatClient,
    system: str,
) -> tuple[str, str, ChatResponse]:
    prompt = (
        raw_history_prompt(entry)
        if system == "raw-history"
        else no_memory_prompt(entry)
    )
    response = reader.complete(prompt, max_tokens=800)
    return response.content, prompt, response


def answer_case(
    entry: dict[str, Any],
    store: ParquetStore,
    reader: ChatClient,
    *,
    top_k: int,
    threshold: float,
    include_expired: bool,
    strategy: str = "cosine",
    pool_size: int = ranking.DEFAULT_POOL_SIZE,
    rrf_k: int = ranking.RRF_K,
) -> tuple[str, list[dict[str, Any]], str, ChatResponse | None]:
    memories = retrieve_memories(
        store,
        entry["question"],
        k=top_k,
        threshold=threshold,
        include_expired=include_expired,
        strategy=strategy,
        pool_size=pool_size,
        rrf_k=rrf_k,
    )
    prompt = answer_prompt(entry, memories)
    if not memories:
        return (
            "I cannot determine the answer from the available memories.",
            memories,
            prompt,
            None,
        )
    response = reader.complete(prompt, max_tokens=800)
    return response.content, memories, prompt, response


def official_judge_prompt(entry: dict[str, Any], hypothesis: str) -> str:
    # Prompt templates follow the MIT-licensed official evaluator:
    # github.com/xiaowu0162/LongMemEval/src/evaluation/evaluate_qa.py
    task = entry["question_type"]
    question = entry["question"]
    answer = entry["answer"]
    if entry["question_id"].endswith("_abs"):
        template = (
            "I will give you an unanswerable question, an explanation, and a "
            "response from a model. Please answer yes if the model correctly "
            "identifies the question as unanswerable. The model could say that "
            "the information is incomplete, or some other information is given "
            "but the asked information is not.\n\nQuestion: {}\n\nExplanation: "
            "{}\n\nModel Response: {}\n\nDoes the model correctly identify the "
            "question as unanswerable? Answer yes or no only."
        )
        return template.format(question, answer, hypothesis)
    if task == "single-session-preference":
        template = (
            "I will give you a question, a rubric for desired personalized "
            "response, and a response from a model. Please answer yes if the "
            "response satisfies the desired response. Otherwise, answer no. The "
            "model does not need to reflect all the points in the rubric. The "
            "response is correct as long as it recalls and utilizes the user's "
            "personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\n"
            "Model Response: {}\n\nIs the model response correct? Answer yes or "
            "no only."
        )
        return template.format(question, answer, hypothesis)
    if task == "temporal-reasoning":
        template = (
            "I will give you a question, a correct answer, and a response from a "
            "model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response is equivalent to the "
            "correct answer or contains all the intermediate steps to get the "
            "correct answer, you should also answer yes. If the response only "
            "contains a subset of the information required by the answer, answer "
            "no. In addition, do not penalize off-by-one errors for the number of "
            "days. If the question asks for the number of days/weeks/months, "
            "etc., and the model makes off-by-one errors (e.g., predicting 19 "
            "days when the answer is 18), the model's response is still correct."
            "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs "
            "the model response correct? Answer yes or no only."
        )
        return template.format(question, answer, hypothesis)
    if task == "knowledge-update":
        template = (
            "I will give you a question, a correct answer, and a response from a "
            "model. Please answer yes if the response contains the correct "
            "answer. Otherwise, answer no. If the response contains some previous "
            "information along with an updated answer, the response should be "
            "considered as correct as long as the updated answer is the required "
            "answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: "
            "{}\n\nIs the model response correct? Answer yes or no only."
        )
        return template.format(question, answer, hypothesis)
    template = (
        "I will give you a question, a correct answer, and a response from a "
        "model. Please answer yes if the response contains the correct answer. "
        "Otherwise, answer no. If the response is equivalent to the correct "
        "answer or contains all the intermediate steps to get the correct "
        "answer, you should also answer yes. If the response only contains a "
        "subset of the information required by the answer, answer no.\n\n"
        "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the "
        "model response correct? Answer yes or no only."
    )
    return template.format(question, answer, hypothesis)


def evaluate_cases(
    entries: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    judge: ChatClient,
    cases_dir: Path | None = None,
) -> dict[str, Any]:
    labels = []
    by_type: dict[str, list[bool]] = defaultdict(list)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
    new_usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
    entry_by_id = {entry["question_id"]: entry for entry in entries}

    for question_id, case in cases.items():
        entry = entry_by_id[question_id]
        existing = case.get("evaluation")
        if existing and existing.get("model") == judge.model:
            label = bool(existing["label"])
            existing_usage = existing.get("usage", {})
            usage["prompt_tokens"] += int(existing_usage.get("prompt_tokens", 0))
            usage["completion_tokens"] += int(
                existing_usage.get("completion_tokens", 0)
            )
            usage["requests"] += 1
        else:
            response = judge.complete(
                official_judge_prompt(entry, case["hypothesis"]),
                max_tokens=200,
            )
            add_usage(usage, response)
            add_usage(new_usage, response)
            label = "yes" in response.content.casefold()
            case["evaluation"] = {
                "model": judge.model,
                "response": response.content,
                "label": label,
                "usage": usage_dict(response),
            }
            if cases_dir is not None:
                atomic_write_json(cases_dir / f"{question_id}.json", case)
        labels.append(label)
        by_type[question_type(entry)].append(label)

    return {
        "evaluated": len(labels),
        "accuracy": round(sum(labels) / len(labels), 4) if labels else 0.0,
        "by_type": {
            name: {
                "accuracy": round(sum(values) / len(values), 4),
                "count": len(values),
            }
            for name, values in sorted(by_type.items())
        },
        "usage": usage,
        "new_usage": new_usage,
    }


def estimate(
    entries: list[dict[str, Any]],
    system: str,
    ingestion_mode: str,
) -> dict[str, Any]:
    unique_sessions: dict[str, int] = {}
    sessions = 0
    turns = 0
    for entry in entries:
        for date, session in zip(entry["haystack_dates"], entry["haystack_sessions"]):
            sessions += 1
            turns += len(session)
            serialized = json.dumps(
                {"date": date, "session": clean_session(session)},
                sort_keys=True,
            )
            digest = hashlib.sha256(serialized.encode()).hexdigest()
            unique_sessions[digest] = len(serialized)
    minimum_requests = len(entries) * 2
    if system == "context-memory" and ingestion_mode == "llm":
        minimum_requests += len(unique_sessions)
    return {
        "system": system,
        "ingestion_mode": ingestion_mode,
        "instances": len(entries),
        "sessions": sessions,
        "unique_sessions": len(unique_sessions),
        "turns": turns,
        "approx_unique_source_tokens": round(sum(unique_sessions.values()) / 4),
        "minimum_llm_requests": minimum_requests,
    }


def aggregate_case_usage(cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    extraction_total = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "requests": 0,
    }
    answer_total = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "requests": 0,
    }
    for case in cases.values():
        extraction = case.get("usage", {}).get("extraction", {})
        extraction_total["prompt_tokens"] += int(
            extraction.get("prompt_tokens", 0)
        )
        extraction_total["completion_tokens"] += int(
            extraction.get("completion_tokens", 0)
        )
        extraction_total["requests"] += int(extraction.get("requests", 0))
        answer = case.get("usage", {}).get("answer", {})
        if answer:
            answer_total["prompt_tokens"] += int(answer.get("prompt_tokens", 0))
            answer_total["completion_tokens"] += int(
                answer.get("completion_tokens", 0)
            )
            answer_total["requests"] += 1
    total = {
        key: extraction_total[key] + answer_total[key]
        for key in extraction_total
    }
    return {
        "extraction": extraction_total,
        "answer": answer_total,
        "total": total,
    }


def reset_case_store(store_path: Path, run_dir: Path) -> None:
    resolved_store = store_path.resolve()
    resolved_run = run_dir.resolve()
    if not resolved_store.is_relative_to(resolved_run):
        raise RuntimeError(f"refusing to reset store outside run directory: {store_path}")
    if store_path.exists():
        shutil.rmtree(store_path)


def build_clients(args: argparse.Namespace) -> tuple[ChatClient, ChatClient, ChatClient]:
    api_key = os.getenv(args.api_key_env, "")
    is_local = "localhost" in args.base_url or "127.0.0.1" in args.base_url
    if not api_key and not is_local:
        raise RuntimeError(
            f"{args.api_key_env} is not set. Set it in PowerShell before running."
        )
    def model_effort(model: str) -> str | None:
        normalized = model.casefold()
        if is_local:
            return args.reasoning_effort
        if normalized.startswith("gpt-5") or normalized.startswith("o"):
            return args.reasoning_effort
        return None

    extractor_model = args.extractor_model or args.reader_model
    judge_model = args.judge_model or args.reader_model
    reader = ChatClient(
        args.base_url,
        api_key,
        args.reader_model,
        reasoning_effort=model_effort(args.reader_model),
    )
    extractor = ChatClient(
        args.base_url,
        api_key,
        extractor_model,
        reasoning_effort=model_effort(extractor_model),
    )
    judge = ChatClient(
        args.base_url,
        api_key,
        judge_model,
        reasoning_effort=model_effort(judge_model),
    )
    return reader, extractor, judge


def main() -> None:
    args = parse_args()
    os.environ["URDWELL_EMBEDDING_BACKEND"] = args.backend
    entries = json.loads(args.dataset.read_text(encoding="utf-8"))
    entries = select_entries(entries, args.limit, args.seed)

    workload = estimate(entries, args.system, args.ingestion_mode)
    print("Workload:", json.dumps(workload, indent=2))
    if args.estimate_only:
        return
    if (
        args.system == "context-memory"
        and args.ingestion_mode == "llm"
        and not args.reuse_stores_from
        and workload["unique_sessions"] > 5000
        and not args.allow_large_llm_ingestion
    ):
        raise RuntimeError(
            "large LLM ingestion is disabled by default because this run would "
            f"extract {workload['unique_sessions']} sessions. Use "
            "--ingestion-mode verbatim, or explicitly add "
            "--allow-large-llm-ingestion."
        )

    reader, extractor, judge = build_clients(args)
    run_dir = DEFAULT_RUNS_DIR / args.run_name
    cases_dir = run_dir / "cases"
    stores_dir = run_dir / "stores"
    cache_dir = DEFAULT_RUNS_DIR / "_extraction_cache"
    reuse_stores_dir = (
        DEFAULT_RUNS_DIR / args.reuse_stores_from / "stores"
        if args.reuse_stores_from
        else None
    )
    if reuse_stores_dir is not None and not reuse_stores_dir.is_dir():
        raise RuntimeError(
            f"--reuse-stores-from: no stores directory at {reuse_stores_dir}"
        )
    existing_cases = list(cases_dir.glob("*.json")) if cases_dir.exists() else []
    if existing_cases and not args.resume:
        raise RuntimeError(
            f"run {args.run_name!r} already contains results; use --resume "
            "or choose another --run-name"
        )
    cases_dir.mkdir(parents=True, exist_ok=True)
    stores_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    completed: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries, start=1):
        question_id = entry["question_id"]
        case_path = cases_dir / f"{question_id}.json"
        if args.resume and case_path.exists():
            case = json.loads(case_path.read_text(encoding="utf-8"))
            if case.get("status") == "complete":
                completed[question_id] = case
                print(f"[{index}/{len(entries)}] resume {question_id}")
                continue

        print(f"[{index}/{len(entries)}] run {args.system} {question_id}")
        case_started = time.perf_counter()
        try:
            if args.system == "context-memory":
                if reuse_stores_dir is not None:
                    # Reuse the memories extracted by a previous run; only the
                    # retrieval/answer/judge stages run again.
                    store_path = reuse_stores_dir / question_id
                    if not store_path.is_dir():
                        raise RuntimeError(
                            f"no reusable store for {question_id} at {store_path}"
                        )
                    store = ParquetStore(store_path)
                    ingestion = []
                    extraction_usage = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "requests": 0,
                    }
                else:
                    store_path = stores_dir / question_id
                    reset_case_store(store_path, run_dir)
                    store = ParquetStore(store_path)
                    if args.ingestion_mode == "llm":
                        ingestion, extraction_usage = ingest_case(
                            entry,
                            store,
                            extractor,
                            cache_dir,
                        )
                    else:
                        ingestion, extraction_usage = ingest_verbatim_case(
                            entry,
                            store,
                        )
                hypothesis, retrieved, prompt, answer_response = answer_case(
                    entry,
                    store,
                    reader,
                    top_k=args.top_k,
                    threshold=args.threshold,
                    include_expired=args.include_expired,
                    strategy=args.ranking,
                    pool_size=args.pool_size,
                    rrf_k=args.rrf_k,
                )
                memory_counts = {
                    "active": len(store.all(active_only=True)),
                    "total": len(store.all(active_only=False)),
                }
            else:
                hypothesis, prompt, answer_response = (
                    answer_without_context_memory(
                        entry,
                        reader,
                        args.system,
                    )
                )
                retrieved = []
                ingestion = []
                extraction_usage = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "requests": 0,
                }
                memory_counts = {"active": 0, "total": 0}
            case = {
                "status": "complete",
                "system": args.system,
                "question_id": question_id,
                "question_type": question_type(entry),
                "question": entry["question"],
                "reference_answer": entry["answer"],
                "question_date": entry["question_date"],
                "hypothesis": hypothesis,
                "retrieved_memories": retrieved,
                "answer_prompt": prompt,
                "ingestion": ingestion,
                "memory_counts": memory_counts,
                "usage": {
                    "extraction": extraction_usage,
                    "answer": (
                        usage_dict(answer_response) if answer_response else {}
                    ),
                },
                "elapsed_seconds": round(time.perf_counter() - case_started, 3),
            }
            atomic_write_json(case_path, case)
            completed[question_id] = case
            elapsed = time.perf_counter() - started
            average = elapsed / index
            eta = average * (len(entries) - index)
            print(
                f"  complete in {case['elapsed_seconds']:.1f}s; "
                f"average {average:.1f}s; ETA {eta / 3600:.2f}h"
            )
            if args.verbose:
                print(json.dumps({"hypothesis": hypothesis}, ensure_ascii=False))
        except Exception as error:
            failure = {
                "status": "error",
                "question_id": question_id,
                "error": repr(error),
            }
            atomic_write_json(case_path, failure)
            print(f"  ERROR: {error}", file=sys.stderr)

    hypotheses_path = run_dir / "hypotheses.jsonl"
    with hypotheses_path.open("w", encoding="utf-8") as output:
        for entry in entries:
            case = completed.get(entry["question_id"])
            if case:
                output.write(
                    json.dumps(
                        {
                            "question_id": entry["question_id"],
                            "hypothesis": case["hypothesis"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    evaluation = None
    if not args.no_evaluate and completed:
        print("Evaluating hypotheses with the official LongMemEval rubric...")
        evaluation = evaluate_cases(entries, completed, judge, cases_dir)
        atomic_write_json(run_dir / "evaluation.json", evaluation)

    summary = {
        "run_name": args.run_name,
        "dataset": str(args.dataset),
        "configuration": {
            "system": args.system,
            "ingestion_mode": args.ingestion_mode,
            "reader_model": reader.model,
            "extractor_model": extractor.model,
            "judge_model": judge.model,
            "base_url": args.base_url,
            "embedding_backend": args.backend,
            "ranking": args.ranking,
            "pool_size": args.pool_size,
            "rrf_k": args.rrf_k,
            "reuse_stores_from": args.reuse_stores_from,
            "top_k": args.top_k,
            "threshold": args.threshold,
            "include_expired": args.include_expired,
            "reasoning_effort": args.reasoning_effort,
            "seed": args.seed,
        },
        "workload": workload,
        "completed": len(completed),
        "failed": len(entries) - len(completed),
        "generation_usage": aggregate_case_usage(completed),
        "evaluation": evaluation,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    atomic_write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
