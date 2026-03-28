from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(log_path: Path, logger_name: str) -> logging.Logger:
    """Configura logger com arquivo e terminal.

    Parametros:
        log_path: Caminho do arquivo de log.
        logger_name: Nome interno do logger.

    Retorno:
        logging.Logger: Logger configurado para a run.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
