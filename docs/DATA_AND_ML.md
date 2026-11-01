# Data & ML Spec — Rabbit Hole Finder

This is the heart of the project. Read it fully before Phase 1.

## 1. Data sources

### 1.1 Wikipedia Clickstream (primary signal)
- Monthly dumps from Wikimedia: `https://dumps.wikimedia.org/other/clickstream/<YYYY-MM>/clickstream-enwiki-<YYYY-MM>.tsv.gz`
- Format: TSV, no header, columns `prev, curr, type, n`.
  - `prev`: referring article title, or a special source: `other-search`, `other-internal`, `other-external`, `other-empty`, `other-other`.
  - `curr`: requested article title.
  - `type`: `link` (prev links to curr), `external`, `other`.
  - `n`: number of occurrences in the month (pairs with n ≤ 10 are dropped by Wikimedia).
- Titles use underscores; normalize to one canonical form and map to integer `article_id`.
- **Discover the latest available months programmatically** by listing the directory; do not hardcode. Default to the 3 most recent complete months. Check file size before download (CLAUDE.md 2 GB rule).
- Parse with DuckDB `read_csv` directly on the gz file; write Parquet.

Derived per-article stats (no separate pageviews dataset needed):
- `views_in` = Σ n over all rows where curr = article
- `onward_clicks` = Σ n over `link` rows where prev = article
- `onward_ratio` = onward_clicks / views_in (how often readers keep clicking; clip to [0, 5])
- `search_entry_share` = n from `other-search` / views_in

Edge table: only `type = 'link'` rows where both ends are in the article subset. `p(src→dst) = n / Σ n out of src` (normalized over link clicks).

### 1.2 Article text
- **DEV mode:** fetch lead sections via the MediaWiki Action API (`action=query&prop=extracts&exintro=1&explaintext=1`, ≤ 20 titles per request, descriptive `User-Agent`, ≤ 1 request/second, honour `maxlag`). Cache responses to disk.
- **FULL mode:** stream the Hugging Face `wikimedia/wikipedia` dataset (pick the most recent English snapshot config available) with `datasets` streaming, keep only articles in the subset, keep the lead (text before the first section heading, truncated to ~2,000 chars) plus the full text for chunking up to ~20,000 chars. Snapshot dates will not match clickstream months exactly; drop or flag titles that fail to match (track match rate, target ≥ 90%). Resolve redirects where possible.

### 1.3 Article subset
- **DEV (~5,000 articles):** pick 25 diverse seed articles (history, science, music, games, sports, food, programming, cricket, mythology, etc.), BFS over clickstream link edges by highest n until ~5,000 articles. Deterministic given the seed list (store it in `backend/src/rabbithole/ingest/dev_seeds.py`).
- **FULL:** articles with `views_in ≥ 1,000` in the feature month (the oldest downloaded month; later months are LTR label months, see DECISIONS.md), excluding `Main_Page`, disambiguation pages, lists (`List_of_*`) and date/year pages. Expect ~100k–300k articles; log the count. Make the threshold configurable.

### 1.4 Hardware assumption
Typical student laptop: 16 GB RAM, no GPU. Everything must run on CPU. Embedding FULL mode may take an hour on CPU; allow an optional path to run the embedding step on a free Colab/Kaggle GPU (export a Parquet of texts, import a Parquet of vectors).

## 2. Rabbit holes = communities

1. Build an undirected weighted graph from link edges: `w(a,b) = n(a→b) + n(b→a)`.
2. Run **Leiden** (igraph `community_leiden`, objective `modularity`, tune `resolution`).
3. Choose resolution by a small sweep: target median community size 20–200 articles for FULL; report modularity, size distribution and coherence (mean intra-community embedding cosine) for each value. Record the choice in DECISIONS.md.
4. Drop communities with < 8 articles from recommendation (keep them in the graph).
5. **Labeling:** c-TF-IDF over member lead texts for keywords; the label is the title of the most central article (highest summed transition probability `p` from other members; raw weighted in-degree let single huge edges dominate, see DECISIONS.md) plus top keywords. Optionally generate a 3–6 word catchy `hook` with the LLM in a batch job (cached; template fallback = keywords).
6. Sanity check: print 20 random communities with top 10 titles into `reports/communities_sample.md`. They must look coherent to a human.

Compare against HDBSCAN on embeddings in a notebook (briefly): which gives more coherent, more "rabbit-holey" groups? Report it; it is a good interview story.

## 3. Rabbit-holeness score (the novel part)

A rabbit hole is a topic where readers who enter **keep going and stay inside**. Clickstream lets us measure exactly that.

### 3.1 Features per community C
| Feature | Definition | Intuition |
|---|---|---|
| `expected_depth` | Expected number of clicks a reader takes inside C before leaving, from an absorbing Markov chain (below) | Core "how deep does it go" |
| `stickiness` | Σ link clicks with both ends in C / Σ link clicks out of members of C | Do people stay? |
| `onward_ratio` | views-weighted mean onward_ratio of members | Do people keep clicking at all? |
| `breadth` | number of articles, log-scaled | Room to explore |
| `content_richness` | median lead+body length of members, log-scaled | Enough to read |
| `discovery_share` | share of entries into C coming from `other-internal`/link vs `other-search` | Found by wandering, not by searching = classic rabbit hole |
| `popularity` | log Σ views_in | Used for baselines/ranking, **not** in the score |

### 3.2 Expected depth (absorbing Markov chain)
For community C with members 1..m, let Q be the m×m matrix of **per-visit** transition probabilities between members, `Q[i,j] = n(i→j) / (max(views_in_i, onward_clicks_i) + 1)`. The missing probability mass of each row is "leave C" (absorbing: clicking outside C *or stopping*). Note this is **not** the edge table's `p`: `p` is normalized over link clicks within the subset, so it ignores stopping and would make every reader click forever (see DECISIONS.md, Phase 4). The `+1` keeps each row sum strictly below 1, so I − Q is always invertible. Fundamental matrix N = (I − Q)⁻¹. Expected pages viewed inside C before absorption, from start vector π (entry distribution into C: each member's views minus visits arriving from other members), is `π · N · 1`.
- Use sparse matrices; solve `(I − Q) y = 1` with a sparse solver rather than inverting, then take `π · y` (equivalent to solving `(I − Q)ᵀ x = π` and summing x).
- Unit test on a 3-node hand-computed example.

### 3.3 Score
`rh_raw = mean of z-scores of [expected_depth, stickiness, onward_ratio, breadth, content_richness, discovery_share]` with weights from config (default equal, `expected_depth` weight 2). `rh_score` = percentile rank of `rh_raw` × 100 → the Depth Meter. Store the feature breakdown in `rh_features` for the UI.

### 3.4 Validating the score (no ground truth, so triangulate)
- Correlation matrix of features; drop or merge features with |r| > 0.85.
- **Face validity:** top 20 and bottom 20 communities by score into `reports/rh_extremes.md`. Top should look like classic rabbit holes (e.g. SCP lore, shipwrecks, conlangs); bottom should look like shallow clusters (e.g. TV episode lists).
- **Human labels:** a small labeling script/CLI where the user rates 100 random communities 1–5 for "would this be a rabbit hole?" Report Spearman ρ between `rh_score` and labels; optionally learn weights with ridge regression on these labels (cross-validated) and compare.

## 4. Recommender

### 4.1 Seeds from free text
Embed each interest string (use the model's query prefix if it has one), nearest articles by cosine in pgvector, take top 3 per interest with a minimum similarity threshold. User seed set S, seed communities C_S.

### 4.2 Candidates
All eligible communities excluding C_S. Precompute per candidate:
- `sem_dist` = 1 − cosine(user centroid, community centroid)
- `graph_dist` = shortest weighted path from any seed to any member on the pruned graph (edge cost = −log p), capped at 6 hops; unreachable = cap + 1.
- `bridge_strength` = max path probability (exp(−cost))
- `rh_score`, `popularity`

For latency, first shortlist the 200 communities nearest by `sem_dist`, then compute graph features only for those.

### 4.3 Heuristic ranker (v1, the explainable baseline of the project)
Goldilocks band: target distance `d* = 0.25 + 0.5·surprise` (in normalized `sem_dist` percentile units), `g(d) = exp(−((d − d*)²) / (2σ²))`, σ = 0.12.
`score = g(sem_dist_pct) · (0.5 + rh_score/200) · (0.7 + 0.3·bridge_strength_pct)`.
Then **MMR re-ranking** (λ = 0.7) on community centroids so the 5 results are diverse.

### 4.4 Learning-to-rank (v2)
Proxy label from **future** reader behaviour, with a temporal split to avoid leakage:
- Feature month(s): M (graph, communities, features built only from M).
- Label months: M+1, M+2.
- For each sampled seed article s (stratified across communities), label each candidate community C by the probability mass of a 3-step random walk from s on the M+1..M+2 graph that lands in C, **excluding mass reached in 1 hop** (we reward discovery, not direct links). Discretize to graded relevance 0–3 by quantiles within each query.
- Train LightGBM `lambdarank` on features from §4.2 + `rh_score` features + surprise-independent features. Group = query (seed set). Split queries by seed community into train/valid/test (no community leakage).
- At serving time, combine: `final = LTR score` reweighted by the Goldilocks band for the user's surprise setting. Document clearly that LTR learns "what readers actually wander into", the band adds user control.
- Track everything in MLflow; register the best model; the API loads the model file path from config.

Be honest in RESULTS.md about the label being a proxy.

### 4.5 Bridge path explanation
Return the most probable path (≤ 4 hops) from a seed to the community's entry article, as titles + URLs: "You like *Rust (programming language)* → readers go to *Memory safety* → *Formal verification* → **Proof assistants rabbit hole**." Paths come only from the graph; never from the LLM.

## 5. RAG trail

1. **Retrieve:** candidate articles = top 30 articles of the community by (centrality, views), plus bridge-path articles. Query = user interests + community label. Retrieve top 12 chunks from `chunks` restricted to those articles (pgvector cosine), max 2 chunks per article.
2. **Order:** gentle → deep using a "gentleness" heuristic: higher views and higher search_entry_share = gentler; more central-but-less-viewed = deeper. Provide this order to the LLM as a suggestion.
3. **Generate:** the LLM returns JSON `{steps: [{article_id, why}], intro}` with 5–7 steps. The prompt includes chunk ids and article ids only; the LLM must choose from them. Instruct it to base each `why` on the given chunks only.
4. **Validate:** reject/repair any step whose `article_id` is not in the retrieved set; URLs are attached by our code from the articles table. If < 5 valid steps, fall back to the template trail.
5. **Cache** by `(community_id, sorted seed ids, model)`.

### RAG evaluation
- Link validity: must be 100% (assert in tests).
- Faithfulness: LLM-as-judge (separate prompt) scoring whether each `why` is supported by its chunk; then the user hand-checks 30 trails to calibrate the judge. Report agreement.
- Ordering: Kendall τ between the trail order and the gentleness ordering (sanity, not a target).

## 6. Offline evaluation (`make eval`)

### 6.1 Protocol
- Queries: 500 seed sets built from held-out seed articles (1–3 articles each from the same or neighbouring communities).
- Ground truth: the future-navigation labels from §4.4 (test split only).

### 6.2 Systems compared
1. Random (eligible communities)
2. Popularity (top by views, excluding seed communities)
3. Nearest neighbour (closest centroids, no rabbit-holeness, no band)
4. Heuristic v1 (no MMR)
5. Heuristic v1 + MMR
6. LTR v2
Ablations: v1 without `rh_score`; v1 without `bridge_strength`.

### 6.3 Metrics
- Accuracy: NDCG@5, Recall@10, MRR
- Beyond accuracy: **novelty** (mean −log₂ popularity share of recommended communities), **intra-list diversity** (mean pairwise centroid distance), **catalog coverage** (share of communities ever recommended across queries), **serendipity** (relevant AND not in the nearest-neighbour baseline's top 10).
- Report mean with 95% bootstrap confidence intervals; paired bootstrap test between v1 and v2.

### 6.4 Outputs
- `reports/eval_<date>.md` + PNG charts (accuracy vs novelty scatter of all systems is the money chart).
- Same metrics logged to MLflow.
- Summarize in `docs/RESULTS.md` and the README.

## 7. Human study (after deploy)
Feedback table already captures ratings. Add `/study` page: participant enters interests, sees 5 recommendations (randomly v1 or v2, blind), rates each 👍/👎/🤯 and novelty 1–5. Report results by system in RESULTS.md. Target 10–20 participants (classmates are fine). No personal data collected.

## 8. Known pitfalls to handle
- Clickstream drops pairs with n ≤ 10: graph is sparse for the long tail. Expect it; note it.
- Redirects and title changes between months: normalize, use the latest month's titles as canonical.
- Hub articles (countries, years, "United States") connect everything: down-weight hubs in path search (add cost proportional to log degree) and exclude them as bridge endpoints.
- Leakage: never compute features for LTR from label months.
- Popularity bias: always report novelty next to accuracy.
