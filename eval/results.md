# Evaluation results (2026-11-16)

## Retrieval (top 5 jobs)

| Questions | Hit rate | Recall | MRR |
|---|---|---|---|
| all 21 answerable | 100% | 98% | 1.00 |

Questions with filters (4), with and without the filters:

| Run | Hit rate | Recall | MRR |
|---|---|---|---|
| with filters | 100% | 100% | 1.00 |
| without filters | 100% | 100% | 0.83 |

Questions where not every expected job was in the top results:

- `fastapi`: missing alpenglow-ai, kranich-ai-agents (got nordlicht-backend, kiezlabs-fullstack, elster-integration, brueckenbau-data, polaris-security)

## Answers (gemini-3.5-flash-lite: 24)

| Check | Result |
|---|---|
| cites a correct job | 21/21 |
| no invented citations | 24/24 |
| declines unanswerable questions | 3/3 |
