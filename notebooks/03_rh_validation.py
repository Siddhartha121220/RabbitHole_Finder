import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from scipy.stats import spearmanr
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict

    from rabbithole.config import REPO_ROOT, get_settings
    from rabbithole.pipelines.definitions import processed
    from rabbithole.scoring.features import FEATURES
    from rabbithole.scoring.labels import read_labels

    settings = get_settings()
    FIG = REPO_ROOT / "reports" / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    scores = pl.read_parquet(processed(settings, "rh_scores"))
    labels = read_labels(settings.data_dir / "labels" / "rh_labels.csv")
    mo.md(
        "# Validating the rabbit-holeness score\n\n"
        'There is no ground truth for "rabbit-holeness", so we triangulate: feature '
        "correlations, face validity (`reports/rh_extremes.md`) and human labels."
    )
    return (
        FEATURES,
        FIG,
        KFold,
        RidgeCV,
        cross_val_predict,
        labels,
        mo,
        np,
        pl,
        plt,
        scores,
        spearmanr,
    )


@app.cell
def _(FEATURES, FIG, mo, np, plt, scores):
    cols = [*FEATURES, "popularity"]
    corr = np.corrcoef(scores.select(cols).to_numpy().T)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    ax.set_title("Feature correlations (eligible communities)")
    fig.tight_layout()
    fig.savefig(FIG / "03_rh_feature_correlation.png", dpi=120)
    off_diag = np.abs(corr[: len(FEATURES), : len(FEATURES)] - np.eye(len(FEATURES)))
    mo.vstack(
        [
            fig,
            mo.md(
                f"Largest |r| between score features: **{off_diag.max():.2f}** "
                "(the spec drops or merges features above 0.85). Popularity is shown for "
                "reference only; it is not part of the score."
            ),
        ]
    )
    return


@app.cell
def _(labels, mo, pl, scores, spearmanr):
    joined = scores.join(labels.select("community_id", "rating"), on="community_id")
    if joined.height < 10:
        result = mo.md(
            f"**{joined.height} human labels so far.** Run "
            "`uv --directory backend run rabbithole label-communities --n 100` "
            "to rate communities (~20 minutes), then re-run this notebook."
        )
        rho = None
    else:
        rho, pval = spearmanr(joined["rh_score"], joined["rating"])
        result = mo.md(
            f"Spearman rho between `rh_score` and human ratings: **{rho:.2f}** "
            f"(p = {pval:.3g}, n = {joined.height})."
        )
    result
    return joined, rho


@app.cell
def _(FEATURES, KFold, RidgeCV, cross_val_predict, joined, mo, np, spearmanr):
    if joined.height < 30:
        ridge_out = mo.md("Ridge-learned weights need at least 30 labels.")
    else:
        X = joined.select([f"z_{f}" for f in FEATURES]).to_numpy()
        y = joined["rating"].to_numpy().astype(float)
        model = RidgeCV(alphas=np.logspace(-2, 3, 30))
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        pred = cross_val_predict(model, X, y, cv=cv)
        rho_cv, _ = spearmanr(pred, y)
        weights = model.fit(X, y).coef_
        rows = "\n".join(f"| {f} | {w:+.3f} |" for f, w in zip(FEATURES, weights, strict=True))
        ridge_out = mo.md(
            f"Cross-validated Spearman rho of ridge-learned weights: **{rho_cv:.2f}** "
            "(compare with the hand-set weights above; learned weights only win if this "
            "is clearly higher).\n\n| feature | learned weight |\n|---|---|\n" + rows
        )
    ridge_out
    return


if __name__ == "__main__":
    app.run()
