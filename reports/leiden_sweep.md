# Leiden resolution sweep (DEV)

Modularity is computed at resolution 1 for every row, so rows are comparable. Coherence = mean intra-community cosine similarity of lead-text TF-IDF vectors (communities with >= 8 articles).

| resolution | communities | eligible | median_size | max_size | share_in_eligible | modularity | coherence |
|---|---|---|---|---|---|---|---|
| 1 | 25 | 23 | 197 | 510 | 0.999 | 0.7618 | 0.0584 |
| 2 | 30 | 29 | 107 | 504 | 0.9996 | 0.7254 | 0.07124 |
| 3 | 39 | 37 | 103 | 450 | 0.9984 | 0.6719 | 0.07603 |
| 4 | 46 | 43 | 68 | 434 | 0.9984 | 0.6578 | 0.081 |
| 5 | 58 | 51 | 42 | 434 | 0.995 | 0.6094 | 0.08514 |
| 6 | 67 | 55 | 39 | 434 | 0.9908 | 0.5847 | 0.09003 |
| 8 | 85 | 63 | 32 | 356 | 0.9826 | 0.5431 | 0.09027 |
| 12 | 106 | 74 | 33 | 354 | 0.979 | 0.4753 | 0.09517 |
| 20 | 148 | 97 | 31 | 201 | 0.9716 | 0.4132 | 0.1016 |

**Chosen resolution: 4** (best modularity x coherence with median community size in 20-200).
