"""Learning-to-rank v2 (DATA_AND_ML §4.4): LightGBM lambdarank on the candidate features.

Labels are the future-navigation grades from `eval.labels`; train / valid queries come from
disjoint seed communities, and the test communities are never touched here.
"""

import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from rabbithole.config import Settings
from rabbithole.eval import labels as lab
from rabbithole.recommend import ranker
from rabbithole.recommend.candidates import Candidates, candidates
from rabbithole.recommend.catalog import Catalog

log = logging.getLogger(__name__)

# Surprise-independent features: the user's surprise setting is applied afterwards by the band.
FEATURES = [
    "sem_dist",
    "sem_pct",
    "graph_dist",
    "bridge_strength",
    "bridge_pct",
    "rh_score",
    "log_views",
    "views_pct",
    "n_seeds",
]
PARAMS: dict[str, Any] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [5],
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 0,
}


def features(catalog: Catalog, cands: Candidates, n_seeds: int) -> np.ndarray:
    views = catalog.views[cands.idx]
    cols = [
        cands.sem_dist,
        ranker.percentile(cands.sem_dist),
        cands.graph_dist,
        cands.bridge_strength,
        ranker.percentile(cands.bridge_strength),
        catalog.rh_score[cands.idx],
        np.log1p(views),
        ranker.percentile(views),
        np.full(len(cands.idx), n_seeds),
    ]
    return np.column_stack(cols).astype(np.float32)


def dataset(
    catalog: Catalog, queries: list[lab.Query], settings: Settings
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    groups: list[int] = []
    for q in queries:
        c = candidates(catalog, q.seeds, settings.shortlist_size, settings.max_hops)
        xs.append(features(catalog, c, len(q.seeds)))
        ys.append(np.array([q.relevance[int(i)] for i in catalog.community_ids[c.idx]]))
        groups.append(len(c.idx))
    return np.vstack(xs), np.concatenate(ys), groups


def train(
    catalog: Catalog,
    t: csr_matrix,
    settings: Settings,
    n_train: int = 3000,
    n_valid: int = 600,
) -> tuple[lgb.Booster, dict[str, float]]:
    splits = lab.community_splits(catalog)
    tq = lab.sample_queries(catalog, t, splits["train"], n_train, seed=21)
    vq = lab.sample_queries(catalog, t, splits["valid"], n_valid, seed=22)
    log.info("ltr queries: %d train, %d valid", len(tq), len(vq))
    xt, yt, gt = dataset(catalog, tq, settings)
    xv, yv, gv = dataset(catalog, vq, settings)
    dt = lgb.Dataset(xt, yt, group=gt, feature_name=FEATURES)
    dv = lgb.Dataset(xv, yv, group=gv, reference=dt)
    model = lgb.train(
        PARAMS,
        dt,
        num_boost_round=500,
        valid_sets=[dv],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    info = {
        "train_queries": float(len(tq)),
        "valid_queries": float(len(vq)),
        "best_iteration": float(model.best_iteration),
        "valid_ndcg_at_5": float(model.best_score["valid_0"]["ndcg@5"]),
    }
    return model, info


def model_path(settings: Settings) -> Path:
    return settings.data_dir / "models" / f"ltr_{settings.dataset_mode}.txt"


def load_model(settings: Settings) -> lgb.Booster:
    return lgb.Booster(model_file=str(model_path(settings)))


def scores(model: lgb.Booster, catalog: Catalog, cands: Candidates, n_seeds: int) -> np.ndarray:
    return np.asarray(model.predict(features(catalog, cands, n_seeds)))


def rank(
    model: lgb.Booster,
    catalog: Catalog,
    cands: Candidates,
    n_seeds: int,
    k: int,
    surprise: float = 0.5,
    sigma: float = 0.12,
    lam: float | None = 0.7,
    band: bool = True,
) -> list[int]:
    """LTR score (as a 0..1 percentile) reweighted by the Goldilocks band, then MMR.
    The band only down-weights (x0.5..1), so the surprise slider steers the list without
    overriding what readers actually wander into."""
    s = ranker.percentile(scores(model, catalog, cands, n_seeds)) + 0.05
    if band:
        g = ranker.goldilocks(ranker.percentile(cands.sem_dist), surprise, sigma)
        s = s * (0.5 + 0.5 * g)
    if lam is None:
        return ranker.top_k(s, k)
    return ranker.mmr(s, catalog.centroids[cands.idx], k, lam)


def importances(model: lgb.Booster) -> dict[str, float]:
    gain = model.feature_importance("gain")
    return dict(zip(FEATURES, (gain / gain.sum()).round(3).tolist(), strict=True))


def train_and_save(settings: Settings, processed_dir: Path) -> dict[str, object]:
    """Train, save next to the data, log to MLflow (skipped if it is down)."""
    from rabbithole.recommend.catalog import load

    catalog = load(processed_dir)
    t = lab.transition_matrix(
        pl.read_parquet(processed_dir / "label_edges.parquet"), len(catalog.titles)
    )
    model, info = train(catalog, t, settings)
    path = model_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    imp = importances(model)
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("ltr")
        with mlflow.start_run(run_name=f"lambdarank-{settings.dataset_mode}"):
            mlflow.log_params({**PARAMS, "mode": settings.dataset_mode, "eval_at": "5"})
            mlflow.log_metrics(info)
            mlflow.log_dict(imp, "feature_importance.json")
            mlflow.log_artifact(str(path))
    except Exception as exc:  # tracking is optional
        log.warning("MLflow logging skipped: %s", exc)
    return {**info, "importance": imp, "path": str(path)}
