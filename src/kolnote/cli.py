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
from .server import serve


async def _run(config: Path | None) -> None:
    settings = load_settings(config)
    specs = settings.channel_specs()
    if not specs and settings.wyoming is None:
        raise SystemExit("nothing to run: add [channel], [channels.<name>] or [wyoming] (see configs/)")
    engine = build_stt(settings.stt)
    services = []
    if settings.wyoming is not None:
        from .wyoming_server import WyomingServer

        services.append(WyomingServer(engine, language=settings.language, **settings.wyoming))
    if hasattr(engine, "warm_up"):
        await engine.warm_up()
    await pipeline.run_channels(
        [(build_channel(spec), Policy.from_config(policy)) for spec, policy in specs],
        engine,
        services=services,
        language=settings.language,
        concurrency=settings.concurrency,
        unsupported_text=settings.unsupported_text or None,
        loading_text=settings.loading_text or None,
    )


async def _serve(config: Path | None) -> None:
    await serve(load_settings(config))


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

    sv = sub.add_parser("serve", help="serve the [stt] engine over the OpenAI /v1/audio/transcriptions API")
    sv.add_argument("-c", "--config", type=Path, default=None, help="TOML file (optional; KOLNOTE_* env vars override)")

    bn = sub.add_parser("bench", help="score one or more STT configs on a labeled dataset")
    bn.add_argument("--dataset", type=Path, required=True)
    bn.add_argument("--stt", type=Path, action="append", required=True, help="config file; repeat to compare")
    bn.add_argument("--language", default=None)
    bn.add_argument("--out", type=Path, default=None)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # request URLs can contain bot tokens
    if args.command == "run":
        asyncio.run(_run(args.config))
    elif args.command == "serve":
        try:
            asyncio.run(_serve(args.config))
        except KeyboardInterrupt:
            pass
    else:
        asyncio.run(_bench(args.dataset, args.stt, args.language, args.out))
