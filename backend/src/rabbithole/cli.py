"""`rabbithole` command line. Usage: uv run rabbithole label-communities --n 100"""

import argparse
import textwrap

import polars as pl

from rabbithole.config import get_settings
from rabbithole.pipelines.definitions import processed
from rabbithole.scoring.labels import SCALE, append_label, first_sentence, to_label

PROMPT = (
    f"Would this be a rabbit hole you could fall into? 1 (not at all) - {SCALE} (absolutely), "
    "s skip, q quit: "
)


def label_communities(n: int) -> None:
    """Blind rating: shows what the community is about and its top articles, never the score."""
    settings = get_settings()
    communities = pl.read_parquet(processed(settings, "communities"))
    texts = pl.read_parquet(processed(settings, "article_text"))
    name = dict(zip(texts["article_id"].to_list(), texts["title"].to_list(), strict=True))
    lead = dict(zip(texts["article_id"].to_list(), texts["lead_text"].to_list(), strict=True))
    path = settings.data_dir / "labels" / "rh_labels.csv"
    todo = to_label(communities, path, n)
    print(f"{len(todo)} communities to rate (saved to {path}; safe to quit and resume).\n")
    by_id = {r["community_id"]: r for r in communities.iter_rows(named=True)}
    valid = {str(i) for i in range(1, SCALE + 1)}
    for k, cid in enumerate(todo, 1):
        c = by_id[cid]
        print(f"[{k}/{len(todo)}] {c['label']}  ({c['size']} articles)")
        about = first_sentence(lead[c["central_article_id"]], max_chars=400)
        if about:
            print(
                textwrap.fill(
                    f"About: {about}",
                    width=100,
                    initial_indent="   ",
                    subsequent_indent="          ",
                )
            )
        print("   keywords: " + ", ".join(c["keywords"]))
        print("   Top articles:")
        for a in c["top_article_ids"]:
            title = name[a].replace("_", " ")
            gist = first_sentence(lead[a], max_chars=110)
            print(f"   - {title}" + (f": {gist}" if gist else ""))
        while True:
            answer = input(PROMPT).strip().lower()
            if answer == "q":
                return
            if answer == "s" or answer in valid:
                break
        if answer != "s":
            append_label(path, cid, c["label"], int(answer))
        print()
    print("Done. Thank you!")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rabbithole")
    sub = parser.add_subparsers(dest="command", required=True)
    lab = sub.add_parser(
        "label-communities", help=f"rate communities 1-{SCALE} for the score check"
    )
    lab.add_argument("--n", type=int, default=100, help="total communities to have rated")
    args = parser.parse_args()
    if args.command == "label-communities":
        label_communities(args.n)
