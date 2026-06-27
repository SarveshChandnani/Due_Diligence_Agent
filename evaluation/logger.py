"""
Centralized logger for the evaluation framework.

All modules should import `logger` from this file.
"""

import logging
from logging.handlers import RotatingFileHandler

from evaluation.config import config


def setup_logger() -> logging.Logger:
    """
    Configure and return the application logger.
    """

    logger = logging.getLogger("rag_evaluation")

    # Avoid duplicate handlers if imported multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    # ------------------------------
    # Log Format
    # ------------------------------
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ------------------------------
    # Console Handler
    # ------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # ------------------------------
    # File Handler
    # ------------------------------
    file_handler = RotatingFileHandler(
        filename=config.LOGS_DIR / "evaluation.log",
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=5,
        encoding="utf-8",
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()