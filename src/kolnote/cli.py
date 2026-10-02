from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from . import bench, pipeline
from .config import load_settings
from .policy import Policy
from .registry import build_channel, build_stt


async def _run(config: Path | None) -> None:
    settings = load_settings(config)
    await pipeline.run(
        build_channel(settings.channel),
        build_stt(settings.stt),
        Policy.from_config(settings.policy),
        language=settings.language,
        concurrency=settings.concurrency,
        unsupported_text=settings.unsupported_text or None,
    )


async def _bench(dataset: Path, configs: list[Path], language: str | None, out: Path | None) -> None:
    results = []
    for cfg in configs:
        settings = load_settings(cfg)
        engine = build_stt(settings.stt)
        try:
            result = await bench.run_bench(dataset, engine, language=language or settings.language)
        finally:
            await engine.close()
        results.append((cfg.stem, result))
        print(f"done: {cfg.stem}")
    print(bench.format_summary(results))
    if out:
        out.write_text(
            json.dumps({label: r.to_dict() for label, r in results}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="kolnote")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the bot from a config file")
    run.add_argument("-c", "--config", type=Path, default=None, help="TOML file (optional; KOLNOTE_* env vars override)")

    bn = sub.add_parser("bench", help="score one or more STT configs on a labeled dataset")
    bn.add_argument("--dataset", type=Path, required=True)
    bn.add_argument("--stt", type=Path, action="append", required=True, help="config file; repeat to compare")
    bn.add_argument("--language", default=None)
    bn.add_argument("--out", type=Path, default=None)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "run":
        asyncio.run(_run(args.config))
    else:
        asyncio.run(_bench(args.dataset, args.stt, args.language, args.out))
