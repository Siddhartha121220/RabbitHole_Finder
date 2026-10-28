# PRD — Rabbit Hole Finder

## 1. Problem
Curious people ("nerds") love falling into a new rabbit hole, but discovery is accidental. Recommenders on most platforms optimize for *more of the same*. Nobody optimizes for **the next obsession**: a topic close enough to what you love to feel approachable, far enough to feel like a discovery, and deep enough to keep you busy for weeks.

## 2. Product in one sentence
Tell us 3–5 things you love; we show you the rabbit holes waiting next door, how deep each one goes, the path from your interests into it, and a grounded reading trail to start with.

## 3. Users
- **Primary:** curious students and hobbyists who browse Wikipedia, Reddit, YouTube explainers.
- **Secondary (portfolio audience):** recruiters and interviewers evaluating the author's data science skills. The repo, README, demo and evaluation report are part of the product.

## 4. Core user flow
1. User enters 3–5 interests as free text (e.g. "mechanical keyboards", "Byzantine Empire", "Rust programming").
2. User sets a **Surprise** slider (0 = safe adjacent picks, 1 = weird and wonderful).
3. System returns **5 rabbit holes**, each with:
   - Name and one-line hook
   - **Depth meter** (rabbit-holeness score, 0–100) with a breakdown
   - **Why this?** — the bridge path from one of the user's interests into the rabbit hole, using real reader navigation data ("people who read X often go to Y, then Z")
   - Top 5 articles inside it
4. User clicks a rabbit hole → **Trail**: an ordered list of 5–7 Wikipedia articles from gentle to deep, each with a 1–2 sentence "why read this", generated with RAG over our own article text. Every link is real.
5. User gives feedback: 👍 "going down it" / 👎 "meh" / 🤯 "already obsessed".
6. **Map view:** an interactive 2D map of all rabbit holes, with the user's interests shown as glowing points and recommendations highlighted.

## 5. MVP scope (must have)
- English Wikipedia only.
- Free-text interest input → seed articles (semantic search).
- Rabbit hole detection (topic communities) and rabbit-holeness score.
- Recommendations with surprise control, depth meter, bridge path explanation.
- RAG trail generation with citations; works in a no-LLM fallback mode.
- Interactive map.
- Feedback capture.
- Offline evaluation report with baselines and ablations.
- Deployed public demo + README with results.

## 6. Stretch goals (only after MVP is done)
- **Rabbit hole autopsy:** paste a list of Wikipedia URLs (your browsing trail) and see the path visualized on the map, with the community you fell into.
- **Personal taste model:** use feedback to re-rank for returning users.
- **"Ask this rabbit hole":** small RAG chat scoped to one community.
- Additional languages where clickstream data exists.

## 7. Non-goals
- No user accounts or auth for MVP (anonymous session id in localStorage is fine).
- No scraping of sites that forbid it. Wikipedia data comes from official dumps and the official API only.
- No real-time ingestion; monthly batch refresh is fine.

## 8. Success metrics
**Offline (computed in `make eval`):**
- Recommender beats popularity and nearest-neighbour baselines on NDCG@5 against held-out future clickstream navigation (see `DATA_AND_ML.md` §6).
- Novelty and intra-list diversity higher than nearest-neighbour baseline at comparable NDCG.
- Trail link validity = **100%** (enforced in code).
- Trail citation faithfulness ≥ 90% on a human-checked sample of 30 trails.

**Human study (small, 10–20 friends/classmates):**
- ≥ 40% of recommendations rated 👍 "would go down it".
- Average novelty rating ≥ 3.5 / 5 ("I didn't know about this").

**Engineering:**
- `/recommend` p95 latency < 800 ms (excluding trail generation) on the deployed instance.
- CI green; test coverage ≥ 70% on `backend/src/`.

## 9. Deliverables
1. Working repo with the commands in `CLAUDE.md`.
2. Deployed demo URL.
3. `README.md`: problem, demo GIF, architecture diagram, how the metric works, results table, what failed, how to run.
4. `docs/RESULTS.md`: full evaluation report with charts.
5. `docs/DECISIONS.md`: decision log.
