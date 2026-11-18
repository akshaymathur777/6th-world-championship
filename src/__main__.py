"""Command line entry points."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Local CDC guideline retrieval")
    parser.add_argument(
        "command",
        choices=["ingest", "freeze-eval", "ablate", "export", "demo", "all"],
    )
    parser.add_argument("--force", action="store_true", help="Rebuild the corpus if it already exists")
    args = parser.parse_args()

    if args.command in {"ingest", "all"}:
        from src.ingest import ingest

        ingest(force=args.force)
    if args.command in {"freeze-eval", "all"}:
        from src.freeze_eval import freeze_eval

        freeze_eval()
    if args.command in {"ablate", "all"}:
        from src.ablate import ablate

        ablate()
    if args.command in {"export", "all"}:
        from src.export_yankee import export_winner

        export_winner()
    if args.command in {"demo", "all"}:
        from src.demo import demo

        demo()


if __name__ == "__main__":
    main()
