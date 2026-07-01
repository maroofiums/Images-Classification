from pathlib import Path
import logging

from src.config import LOGS_DIR

# Create Log Directory
LOGS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def get_logger(
    name: str,
    log_file: str | Path,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Create and configure a logger.

    Args:
        name: Logger name.
        log_file: Log filename.
        level: Logging level.
        console: Whether to also log to the console.

    Returns:
        Configured logger.
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(
        LOGS_DIR / log_file,
        encoding="utf-8",
    )

    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.propagate = False

    return logger


# Application Loggers
app_logger = get_logger(
    name="app",
    log_file="app.log",
)

access_logger = get_logger(
    name="access",
    log_file="access.log",
)

error_logger = get_logger(
    name="error",
    log_file="error.log",
    level=logging.ERROR,
)

inference_logger = get_logger(
    name="inference",
    log_file="inference.log",
)