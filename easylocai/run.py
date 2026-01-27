import argparse
import asyncio
import logging
import sys

from easylocai.config import ensure_user_config
from easylocai.main import run_agent_workflow
from easylocai.utlis.loggers.default_dict import default_logging_config

logging.config.dictConfig(default_logging_config)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="easylocai",
        description="Easy Local AI CLI",
    )

    subparsers = parser.add_subparsers(dest="command")

    # ADD easylocai init command
    # easylocai init
    init_parser = subparsers.add_parser(
        "init",
        help="Create config at ~/.config/easylocai/config.json",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config",
    )

    # Global flag option for feature flags
    parser.add_argument(
        "--flag",
        type=str,
        default=None,
        help="Feature flag (e.g., --flag=beta)",
    )
    return parser


def run() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init":
        path = ensure_user_config(overwrite=args.force)
        print(f"Config initialized at: {path}")
        return 0

    ensure_user_config(overwrite=False)

    try:
        asyncio.run(run_agent_workflow(flag=args.flag))
    except KeyboardInterrupt:
        print("\nExiting...")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(run())
