# LongMemEval Benchmark

This directory contains the ContextMemory benchmark harness and locally
downloaded LongMemEval data.

The official cleaned dataset is published by the LongMemEval authors:

https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

Downloaded benchmark JSON files are ignored by Git because they are generated
inputs rather than project source.

## Retrieval Evaluation

Run a quick deterministic smoke test:

```bash
uv run python benchmarks/longmemeval/run_retrieval.py \
  --backend hashing \
  --granularity turn \
  --limit 100
```

Run the production MiniLM retriever:

```bash
uv run python benchmarks/longmemeval/run_retrieval.py \
  --backend fastembed \
  --granularity turn \
  --limit 100
```

By default, the indexed corpus matches the official retrieval baseline and
contains only user text. Add `--include-date` to run the ContextMemory
time-aware variant, which prefixes each indexed item with its session date:

```bash
uv run python benchmarks/longmemeval/run_retrieval.py \
  --backend fastembed \
  --granularity turn \
  --include-date
```

Remove `--limit` to evaluate all 500 instances. Reports are written under
`benchmarks/longmemeval/reports/`.

This harness measures retrieval, not final question-answering accuracy.
ContextMemory does not yet include the extraction and reader LLMs needed for a
complete LongMemEval QA evaluation.

See `RESULTS.md` for the current MiniLM baseline and threshold calibration.

## Ranking Comparison: Cosine vs Hybrid (BM25 + RRF)

ContextMemory ranks retrieved memories one of two ways, selectable with
`--ranking` on the end-to-end runner and used by the production
`search_memory` server tool.

- **`cosine`** — pure dense retrieval. Each memory is scored by the cosine
  similarity between its embedding and the query embedding, and the top *k* win.
  Captures semantic similarity, but can under-rank exact-term matches.
- **`hybrid`** — dense cosine fused with lexical **BM25**.
  - **BM25** is a classic bag-of-words relevance function. It scores a memory by
    how many query *terms* it contains, weighting rare terms more (IDF) and
    saturating repeated terms. It rescues exact matches on names, identifiers,
    and rare words that an embedding smooths over.
  - **RRF (Reciprocal Rank Fusion)** merges the cosine ranking and the BM25
    ranking by summing `1 / (k + rank)` per memory across both lists, without
    comparing their incommensurable raw scores. Hybrid is therefore
    *cosine **plus** BM25, fused by RRF* — not an alternative to cosine. The
    abstention decision stays pure cosine, so specificity is unchanged by
    construction.

### What we measured

Two comparisons on the cleaned Oracle dataset with identical stacks (FastEmbed
multilingual MiniLM, `gemma4:26b` locally via Ollama, top-k 5, threshold 0,
seed 42, `pool_size` 50, `rrf_k` 60) — only the ranking changed.

**1. Retrieval quality** — `run_retrieval.py` vs `run_retrieval_hybrid.py`,
419 answerable questions, no LLM involved:

| Metric (Standard ranking) | Cosine | Hybrid | Δ |
|---|---:|---:|---:|
| recall_any@1 | 0.5513 | 0.6038 | **+5.3 pt** |
| ndcg_any@5 | 0.6966 | 0.7619 | **+6.5 pt** |
| recall_all@5 | 0.6945 | 0.7589 | **+6.4 pt** |
| recall_any@5 | 0.9451 | 0.9666 | +2.2 pt |
| recall_any@10 | 0.9952 | 0.9928 | ≈ (ceiling) |

Per-type NDCG@5 gains concentrate where lexical signal helps most:
temporal-reasoning +10 pt (0.622 → 0.722), multi-session +6.3 pt, and
knowledge-update +5.1 pt.

**2. End-to-end QA accuracy** — `run_end_to_end.py`, 500 cases, local judge:

| | Cosine (`e2e-full`) | Hybrid (`e2e-full-rrf`) |
|---|---:|---:|
| Overall accuracy | 0.528 | 0.528 |
| knowledge-update | 0.792 | 0.861 (+5 cases) |
| temporal-reasoning | 0.402 | 0.417 (+2 cases) |
| multi-session | 0.438 | 0.397 (−5 cases) |
| Runtime | ~60 h (full LLM ingestion) | ~8.8 h (reused stores) |

Hybrid redistributes correct answers (+7 / −7) for **zero net change** in
overall accuracy.

### Conclusion

- **Hybrid (BM25 + RRF) is a clear retrieval win** — better rank-1 and ordering,
  with no measured downside. It is already the default in the production
  `search_memory` server tool.
- **On Oracle, retrieval is not the QA bottleneck.** Better-ranked, more complete
  evidence did not raise answer accuracy; the reader (`gemma4:26b`) is the limit,
  especially for multi-session synthesis. Oracle also under-tests ranking because
  most cases hold ≤ 5 memories — tiny pools leave little to reorder.
- **The next levers are downstream**, not the ranker: the reader model / answer
  prompt, the extraction step (assistant-turn content is under-captured —
  `single-session-assistant` QA ≈ 11 %), and the fixed 0.55 abstention threshold,
  which caps recall for both rankers alike.
- The local `gemma4:26b` judge is for internal comparison only; it is not
  comparable with the official GPT-4o judge.

### Reproducing cheaply

Re-rank an existing run's stores without paying the LLM extraction cost again
(extraction is ~78 % of runtime), via `--reuse-stores-from` and `--ranking`:

```powershell
python benchmarks\longmemeval\run_end_to_end.py `
  --dataset benchmarks\longmemeval\longmemeval_oracle.json `
  --run-name e2e-full-rrf `
  --reader-model gemma4:26b --judge-model gemma4:26b `
  --base-url http://127.0.0.1:11434/v1 --reasoning-effort none `
  --ingestion-mode llm --ranking hybrid --reuse-stores-from e2e-full
```

## End-to-End Evaluation

The end-to-end runner performs the complete memory workflow:

1. replay timestamped sessions into an isolated archive;
2. extract and consolidate structured memories;
3. retrieve memories with ContextMemory;
4. generate an answer;
5. grade it with the official LongMemEval LLM-as-a-judge rubric.

It accepts any provider exposing an OpenAI-compatible
`/v1/chat/completions` endpoint.

## Free Local Run With Ollama

The tested local configuration is:

- Ollama `0.21.2`
- `gemma4:26b` for ContextMemory answers and local judging
- `gemma4-8b-128k` for the raw-history baseline
- MiniLM for semantic retrieval
- no API key and no usage fees

Run three diversified LongMemEval-S cases:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\benchmarks\longmemeval\run_local.ps1 `
  -Dataset s `
  -RunName local-smoke-3 `
  -Limit 3 `
  -VerboseOutput
```

Run all 500 cases:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\benchmarks\longmemeval\run_local.ps1 `
  -Dataset s `
  -RunName local-s-500 `
  -Limit 0
```

Resume the same run after an interruption:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\benchmarks\longmemeval\run_local.ps1 `
  -Dataset s `
  -RunName local-s-500 `
  -Limit 0 `
  -Resume
```

The local script defaults to `verbatim` ingestion. It stores dated
user-assistant rounds without an extraction LLM, then runs ContextMemory
retrieval, answer generation, and local judging. A measured S case with 476
turns took 168 seconds on the reference Zenbook, suggesting roughly 23 hours
for 500 cases. Actual duration varies by case and thermal conditions.

Full LLM extraction is deliberately blocked on S unless
`--allow-large-llm-ingestion` is supplied: the dataset contains 23,867
sessions, and a measured median extraction would make that mode take several
weeks on CPU. Use LLM ingestion on Oracle or a small S sample instead.

The local judge is useful for comparing project variants but is not directly
comparable with published scores produced by the official GPT-4o judge.

### 1. Inspect the workload

This command does not call an API:

```powershell
python benchmarks\longmemeval\run_end_to_end.py `
  --run-name estimate-oracle `
  --reader-model gpt-5.4-mini `
  --estimate-only
```

### 2. Configure the provider

For OpenAI, set the key only in the current PowerShell session:

```powershell
$env:OPENAI_API_KEY = "your-key"
```

For a local OpenAI-compatible server, pass its URL with `--base-url`. A key is
not required for `localhost` or `127.0.0.1`.

### 3. Run a visible three-case smoke test

```powershell
python benchmarks\longmemeval\run_end_to_end.py `
  --run-name smoke-3 `
  --reader-model gpt-5.4-mini `
  --judge-model gpt-4o-2024-08-06 `
  --reasoning-effort low `
  --limit 3 `
  --verbose
```

Inspect the generated files under
`benchmarks/longmemeval/reports/e2e/smoke-3/`. Each JSON file in `cases/`
contains the extracted memories, arbitration decisions, retrieved memories,
answer prompt, hypothesis, and judge decision.

### 4. Run all 500 Oracle cases

```powershell
python benchmarks\longmemeval\run_end_to_end.py `
  --run-name oracle-500 `
  --reader-model gpt-5.4-mini `
  --judge-model gpt-4o-2024-08-06 `
  --reasoning-effort low
```

If the process stops, run the same command with `--resume`.

The Oracle dataset contains only evidence sessions. It validates extraction,
consolidation, temporal updates, reading, and abstention, but it does not stress
retrieval against long distractor histories.

### 5. Run the real LongMemEval-S stress test

Download the cleaned S dataset:

```powershell
python benchmarks\longmemeval\download_dataset.py --variant s
```

Estimate it first:

```powershell
python benchmarks\longmemeval\run_end_to_end.py `
  --dataset benchmarks\longmemeval\longmemeval_s_cleaned.json `
  --run-name estimate-s `
  --reader-model gpt-5.4-mini `
  --estimate-only
```

Then run all 500 cases:

```powershell
python benchmarks\longmemeval\run_end_to_end.py `
  --dataset benchmarks\longmemeval\longmemeval_s_cleaned.json `
  --run-name s-500 `
  --reader-model gpt-5.4-mini `
  --judge-model gpt-4o-2024-08-06 `
  --reasoning-effort low
```

Extraction results are cached across runs for the same model and session. The
final accuracy is written to `summary.json` and `evaluation.json`.

### 6. Compare against the necessary baselines

Run the same reader directly on the complete S history:

```powershell
python benchmarks\longmemeval\run_end_to_end.py `
  --system raw-history `
  --dataset benchmarks\longmemeval\longmemeval_s_cleaned.json `
  --run-name s-500-raw `
  --reader-model gpt-5.4-mini `
  --judge-model gpt-4o-2024-08-06 `
  --reasoning-effort low
```

Optionally measure the no-memory floor:

```powershell
python benchmarks\longmemeval\run_end_to_end.py `
  --system no-memory `
  --dataset benchmarks\longmemeval\longmemeval_s_cleaned.json `
  --run-name s-500-none `
  --reader-model gpt-5.4-mini `
  --judge-model gpt-4o-2024-08-06 `
  --reasoning-effort low
```

Compare the summaries:

```powershell
python benchmarks\longmemeval\compare_runs.py `
  benchmarks\longmemeval\reports\e2e\s-500\summary.json `
  benchmarks\longmemeval\reports\e2e\s-500-raw\summary.json `
  benchmarks\longmemeval\reports\e2e\s-500-none\summary.json
```

ContextMemory demonstrates useful value when it substantially beats the
no-memory floor and approaches raw-history accuracy while sending much less
context to the reader. Pay particular attention to `knowledge-update`,
`multi-session`, and `temporal-reasoning`.

`gpt-4o-2024-08-06` is the judge used by the official evaluator. If that
snapshot is unavailable to the account, use another judge model but treat the
score as internally useful rather than strictly comparable with published
LongMemEval results.
