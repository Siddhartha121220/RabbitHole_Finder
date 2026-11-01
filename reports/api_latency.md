# /recommend latency (DEV, local)

100 random queries (1-3 article titles as interests, random surprise), sequential, end to end over HTTP, encoder on CPU.

| p50 | p95 | p99 | max | non-200 |
|---|---|---|---|---|
| 174 ms | 259 ms | 268 ms | 270 ms | 0 |

Target p95 < 800 ms: **PASS**
