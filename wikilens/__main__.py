"""Run with uv run python -m wikilens."""

import argparse
import json
from pathlib import Path

from .dataset import DEFAULT_DATA, download_dataset, profile_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Wikipedia cross-language evidence explorer"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    for command in ("download", "profile", "load"):
        command_parser = subcommands.add_parser(command)
        command_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    subcommands.add_parser("init-db")
    search = subcommands.add_parser("search")
    search.add_argument("query")
    search.add_argument(
        "--mode", choices=["keyword", "semantic", "hybrid"], default="keyword"
    )
    search.add_argument("--lang", choices=["fi", "en", "sv"])
    compare = subcommands.add_parser("compare")
    compare.add_argument("wikidata_id")
    ask = subcommands.add_parser("ask")
    ask.add_argument("question")
    capture_import = subcommands.add_parser("import-capture")
    capture_import.add_argument("capture", type=Path)
    inspected_run = subcommands.add_parser("inspect-run")
    inspected_run.add_argument("report_hash")
    web = subcommands.add_parser("web")
    web.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "download":
        download_dataset(args.data_dir)
    elif args.command == "profile":
        print(json.dumps(profile_dataset(args.data_dir), indent=2))
    else:
        from .database import compare_topic, initialize, load_dataset, search_articles

        if args.command == "init-db":
            initialize()
        elif args.command == "load":
            print(json.dumps(load_dataset(args.data_dir), indent=2))
        elif args.command == "search":
            print(
                json.dumps(
                    search_articles(args.query, mode=args.mode, lang=args.lang),
                    indent=2,
                    ensure_ascii=False,
                )
            )
        elif args.command == "compare":
            print(
                json.dumps(
                    compare_topic(args.wikidata_id),
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )
        elif args.command == "ask":
            from .rag import answer_question

            print(
                json.dumps(answer_question(args.question), indent=2, ensure_ascii=False)
            )
        elif args.command == "import-capture":
            from .provenance import import_capture

            print(
                json.dumps(
                    import_capture(json.loads(args.capture.read_text())), indent=2
                )
            )
        elif args.command == "inspect-run":
            from .provenance import inspect_run

            result = inspect_run(args.report_hash)
            if result is None:
                raise SystemExit("Reviewed run not found")
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        elif args.command == "web":
            from .web import create_app

            create_app().run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
