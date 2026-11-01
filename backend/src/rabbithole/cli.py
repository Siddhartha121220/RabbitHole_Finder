"""`rabbithole` command line. Usage: uv run rabbithole label-communities --n 100"""

import argparse

import polars as pl

from rabbithole.config import get_settings
from rabbithole.pipelines.definitions import processed
from rabbithole.scoring.labels import append_label, to_label

PROMPT = (
    "Would this be a rabbit hole you could fall into? 1 (no) - 5 (absolutely), s skip, q quit: "
)


def label_communities(n: int) -> None:
    """Blind rating: shows label, keywords and top articles, never the score."""
    settings = get_settings()
    communities = pl.read_parquet(processed(settings, "communities"))
    titles = pl.read_parquet(processed(settings, "article_ids"))
    name = dict(zip(titles["article_id"].to_list(), titles["title"].to_list(), strict=True))
    path = settings.data_dir / "labels" / "rh_labels.csv"
    todo = to_label(communities, path, n)
    print(f"{len(todo)} communities to rate (saved to {path}; safe to quit and resume).\n")
    by_id = {r["community_id"]: r for r in communities.iter_rows(named=True)}
    for k, cid in enumerate(todo, 1):
        c = by_id[cid]
        print(f"[{k}/{len(todo)}] {c['label']}  ({c['size']} articles)")
        print("   keywords: " + ", ".join(c["keywords"]))
        for a in c["top_article_ids"]:
            print("   - " + name[a].replace("_", " "))
        while True:
            answer = input(PROMPT).strip().lower()
            if answer == "q":
                return
            if answer == "s" or answer in {"1", "2", "3", "4", "5"}:
                break
        if answer != "s":
            append_label(path, cid, c["label"], int(answer))
        print()
    print("Done. Thank you!")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rabbithole")
    sub = parser.add_subparsers(dest="command", required=True)
    lab = sub.add_parser("label-communities", help="rate communities 1-5 for the score check")
    lab.add_argument("--n", type=int, default=100, help="total communities to have rated")
    args = parser.parse_args()
    if args.command == "label-communities":
        label_communities(args.n)
