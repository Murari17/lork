"""Zero-ML-import command line entrypoint for lork."""

from __future__ import annotations

import argparse
import os
import sys
import threading

from lork import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lork",
        description="Lork LoRA fine-tuning SDK and async execution server.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"lork {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the lork FastAPI execution server (requires lork[server]).",
    )
    serve_parser.add_argument(
        "--host",
        default=os.environ.get("LORK_HOST", "127.0.0.1"),
        help="Bind host (default: LORK_HOST or 127.0.0.1).",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("LORK_PORT", "8000")),
        help="Bind port (default: LORK_PORT or 8000).",
    )
    serve_parser.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        help="Uvicorn log level (default: info).",
    )
    return parser


def _cmd_serve(host: str, port: int, log_level: str) -> int:
    try:
        import uvicorn
    except ImportError:
        sys.stderr.write(
            'Error: lork server dependencies are not installed. Run: pip install "lork[server]"\n'
        )
        return 2

    try:
        from tqdm import tqdm

        tqdm.set_lock(threading.RLock())
    except ImportError:
        pass

    uvicorn.run(
        "lork.server.app:create_app",
        factory=True,
        host=host,
        port=port,
        log_level=log_level,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `lork`."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        if exc.code is None:
            return 0
        return int(exc.code) if isinstance(exc.code, int) else 1
    if args.command == "serve":
        return _cmd_serve(host=args.host, port=args.port, log_level=args.log_level)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
