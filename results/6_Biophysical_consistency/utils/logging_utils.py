from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_run_logger(
    log_dir: Path,
    run_name: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    创建同时写入文件与标准输出的 Logger。

    日志文件命名：{run_name}_{timestamp}.log
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(run_name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info("Log file: %s", log_path)
    return logger
