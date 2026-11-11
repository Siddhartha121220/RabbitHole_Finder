"""Move a pipeline run between machines (e.g. FULL built on Kaggle, served on a laptop).

    rabbithole pack --out /kaggle/working/rabbithole_full   # on the build machine
    rabbithole unpack ~/Downloads/rabbithole_full            # on the serving machine

Only what serving, LTR and evaluation read is packed. The article bodies (up to 20k chars
each) are dropped: they were only needed to cut chunks, and the chunks travel with their text.
"""

import json
import shutil
from pathlib import Path

import polars as pl

from rabbithole.config import REPO_ROOT, Settings

SLIM = {"article_text": ["body_text"]}  # table -> columns left behind
MANIFEST = "manifest.json"


def pack(settings: Settings, out: Path) -> Path:
    src = settings.data_dir / "processed" / settings.dataset_mode
    dst = out / "processed" / settings.dataset_mode
    dst.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, int]] = {}
    for path in sorted(src.glob("*.parquet")):
        target = dst / path.name
        drop = SLIM.get(path.stem)
        if drop:
            pl.scan_parquet(path).drop(drop).sink_parquet(target)
        else:
            shutil.copy2(path, target)
        rows = pl.scan_parquet(target).select(pl.len()).collect().item()
        files[path.name] = {"rows": rows, "bytes": target.stat().st_size}

    reports = out / "reports"
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    suffix = f"_{settings.dataset_mode}"
    for md in (REPO_ROOT / "reports").glob(f"*{suffix}.md"):
        shutil.copy2(md, reports / md.name)
    for fig in (REPO_ROOT / "reports" / "figures").glob(f"*{suffix}*"):
        shutil.copy2(fig, reports / "figures" / fig.name)

    manifest = {
        "dataset_mode": settings.dataset_mode,
        "full_min_views": settings.full_min_views,
        "clickstream_months": settings.clickstream_months,
        "embedding_model": settings.embedding_model,
        "max_chunks_per_article": settings.max_chunks_per_article,
        "files": files,
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return out


def unpack(settings: Settings, src: Path) -> dict[str, object]:
    """Copy a packed run into data/processed/<mode> and reports/, checking row counts."""
    manifest = json.loads((src / MANIFEST).read_text())
    mode = manifest["dataset_mode"]
    dst = settings.data_dir / "processed" / mode
    dst.mkdir(parents=True, exist_ok=True)
    for name, meta in manifest["files"].items():
        target = dst / name
        shutil.copy2(src / "processed" / mode / name, target)
        rows = pl.scan_parquet(target).select(pl.len()).collect().item()
        if rows != meta["rows"]:
            raise ValueError(f"{name}: {rows} rows, manifest says {meta['rows']}")
    reports = src / "reports"
    if reports.exists():
        shutil.copytree(reports, REPO_ROOT / "reports", dirs_exist_ok=True)
    return manifest
