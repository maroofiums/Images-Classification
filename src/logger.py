from pathlib import Path
import logging

from src.config import LOGS_DIR
import os


LOGS_DIR.mkdir(parents=True,exist_ok=True)

def get_logger(
    name: str,
    log_file: str,
    level = logging.INFO
):

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handle = logging.FileHandler(LOGS_DIR/log_file,encoding="utf-8")

    file_handle.setFormatter(formatter)

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(file_handle)
    logger.addHandler(console_handler)

    logger.propagate = False

    return logger


app_logger = get_logger(
    "app",
    "app.log"
)

access_logger = get_logger(
    "access",
    "access.log"
)

error_logger = get_logger(
    "error",
    "error.log",
    level=logging.ERROR
)

inference_logger = get_logger(
    "inference",
    "inference.log"
)
