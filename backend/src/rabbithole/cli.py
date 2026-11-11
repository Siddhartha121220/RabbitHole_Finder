"""`rabbithole` command line. Usage: uv run rabbithole label-communities --n 100"""

import argparse
import logging
import textwrap
from pathlib import Path

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
    sub.add_parser("eval", help="offline evaluation -> reports/eval_<date>.md + MLflow")
    judge = sub.add_parser("judge-trails", help="LLM-judged trail faithfulness (costs money)")
    judge.add_argument("--n", type=int, default=50, help="trails to generate and judge")
    judge.add_argument("--yes", action="store_true", help="allow more than 200 trails")
    judge.add_argument("--agreement", action="store_true", help="judge vs human hand-checks")
    pack = sub.add_parser("pack", help="bundle processed data + reports to move a run")
    pack.add_argument("--out", type=Path, required=True)
    unpack = sub.add_parser("unpack", help="install a bundle made by `pack`")
    unpack.add_argument("src", type=Path)
    args = parser.parse_args()
    if args.command == "label-communities":
        label_communities(args.n)
    elif args.command == "eval":
        run_eval()
    elif args.command == "pack":
        from rabbithole.pipelines import bundle

        print(f"packed into {bundle.pack(get_settings(), args.out)}")
    elif args.command == "unpack":
        from rabbithole.pipelines import bundle

        m = bundle.unpack(get_settings(), args.src)
        print(f"installed {m['dataset_mode']} run: {len(m['files'])} tables")  # type: ignore[arg-type]
    elif args.command == "judge-trails":
        from rabbithole.eval import trail_judge

        if args.agreement:
            print(trail_judge.agreement())
        else:
            print(f"wrote {trail_judge.run(get_settings(), args.n, args.yes)}")


def run_eval() -> None:
    from rabbithole.eval.runner import run
    from rabbithole.ingest.clickstream import resolve_months

    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    months = resolve_months(settings)
    out = run(settings, processed(settings, "x").parent, months[0], months[1:])
    print(f"wrote {out}")
