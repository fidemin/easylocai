import uuid
from datetime import datetime
from pathlib import Path


def make_logging_config(log_file: str | Path | None = None) -> dict:
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_dir = Path.home() / ".easylocai" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        log_file = log_dir / f"session_{timestamp}_{uid}.log"

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": "DEBUG",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "standard",
                "level": "DEBUG",
                "filename": str(log_file),
                "mode": "w",
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["file"],
                "level": "INFO",
                "propagate": False,
            },
            "easylocai": {
                "handlers": ["file"],
                "level": "DEBUG",
                "propagate": False,
            },
            "__main__": {
                "handlers": ["file"],
                "level": "DEBUG",
                "propagate": False,
            },
        },
    }
