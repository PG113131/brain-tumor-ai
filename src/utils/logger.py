import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "brain_tumor_ai",
    log_dir: str = "logs",
    debug: bool = False,
) -> logging.Logger:
    """
    Configure and return the application logger.
    This module does NOT import settings to avoid circular imports.
    """

    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(
        log_path / "app.log",
        encoding="utf-8",
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger


logger = setup_logger()