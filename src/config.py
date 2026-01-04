import json
from pathlib import Path

DEFAULT_CONFIG = {"mcpServers": {}}


def user_config_path() -> Path:
    return Path.home() / ".config" / "easylocai" / "config.json"


def ensure_user_config(*, overwrite=False) -> Path:
    cfg = user_config_path()
    cfg.parent.mkdir(parents=True, exist_ok=True)

    if overwrite or not cfg.exists():
        cfg.write_text(
            json.dumps(
                DEFAULT_CONFIG,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return cfg
