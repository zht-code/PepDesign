from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_file_logger(log_dir: Path, name: str = "pipeline") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Log file: %s", log_path)
    return logger
