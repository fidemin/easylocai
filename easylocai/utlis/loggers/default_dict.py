default_logging_config = {
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
            "filename": "app.log",
            "mode": "a",
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
