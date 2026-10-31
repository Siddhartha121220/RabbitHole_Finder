# Leiden resolution sweep (DEV)

Modularity is computed at resolution 1 for every row, so rows are comparable. Coherence = mean intra-community cosine similarity of article embeddings (communities with >= 8 articles); coherence_lift = coherence minus the mean cosine of random article pairs. score = modularity x coherence_lift.

| resolution | communities | eligible | median_size | max_size | share_in_eligible | modularity | coherence | coherence_lift | score |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 25 | 23 | 192 | 506 | 0.999 | 0.762 | 0.5789 | 0.1146 | 0.08736 |
| 2 | 31 | 30 | 110 | 503 | 0.9996 | 0.7257 | 0.5967 | 0.1324 | 0.09611 |
| 3 | 40 | 38 | 98.5 | 440 | 0.9986 | 0.6721 | 0.5996 | 0.1354 | 0.09098 |
| 4 | 47 | 44 | 61 | 425 | 0.9984 | 0.6568 | 0.6067 | 0.1425 | 0.09361 |
| 5 | 57 | 50 | 47 | 425 | 0.995 | 0.6028 | 0.6123 | 0.148 | 0.08924 |
| 6 | 66 | 54 | 43 | 424 | 0.9906 | 0.5939 | 0.6137 | 0.1494 | 0.08876 |
| 8 | 86 | 63 | 38 | 385 | 0.9818 | 0.5244 | 0.6205 | 0.1563 | 0.08196 |
| 12 | 107 | 76 | 34.5 | 235 | 0.979 | 0.4801 | 0.6253 | 0.1611 | 0.07735 |
| 20 | 146 | 97 | 31 | 200 | 0.973 | 0.4225 | 0.6315 | 0.1673 | 0.07069 |

**Chosen resolution: 4** (finest resolution within 5% of the best score, median community size in 20-200).
