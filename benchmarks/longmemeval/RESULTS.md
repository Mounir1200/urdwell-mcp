# LongMemEval Retrieval Baseline

Benchmark date: 2026-06-13

## Configuration

- Dataset: official cleaned LongMemEval Oracle, 500 instances
- Retriever: `paraphrase-multilingual-MiniLM-L12-v2`
- Granularity: user turns
- Cutoffs: 1, 5, and 10
- Evaluated: 419 answerable questions
- Excluded by the official user-turn protocol: 30 abstention questions and
  51 questions whose evidence appears only in assistant turns

## Results

| Variant | Recall any @1 | Recall any @5 | Recall all @5 | NDCG @5 |
|---|---:|---:|---:|---:|
| User text only | 0.5513 | 0.9451 | 0.6945 | 0.6972 |
| User text plus session date | 0.5704 | 0.9475 | 0.6921 | 0.7053 |

Adding session dates improves rank-one retrieval, including temporal-reasoning
recall from 0.4409 to 0.4882. It has little effect on top-five recall.

## Threshold Calibration

The current absolute similarity threshold is 0.55. On the undated baseline it
produces:

- Recall any @5: 0.3938
- Abstention specificity: 0.7333
- Balanced score: 0.5636

Among thresholds from 0.20 through 0.60, 0.55 has the best balanced score.
However, no fixed threshold separates relevant evidence from distractors well:
lower thresholds recover answerable cases but return results for nearly every
abstention case.

The next retrieval iteration should therefore keep ranked top-k retrieval and
replace the single absolute cutoff with a learned or query-adaptive confidence
policy.

## Scope

This is a retrieval benchmark, not the complete LongMemEval question-answering
evaluation. UrdWell still needs a conversation-to-memory extractor and a
reader capable of synthesizing answers, resolving updates, and abstaining from
the retrieved evidence.
