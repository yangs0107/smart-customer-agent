"""日志：控制台 + 滚动文件双输出。

Agent 是否调了工具、传了什么参数、命中多少切片，全靠日志。
排查顺序见 SOP「排错优先级」。
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from utils.path_tool import get_abs_path

LOG_DIR = Path(get_abs_path("logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "agent.log"

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _build_logger() -> logging.Logger:
    log = logging.getLogger("agent")
    if log.handlers:  # 防止模块被重复导入时叠加 handler
        return log

    log.setLevel(logging.INFO)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    log.addHandler(console)
    log.addHandler(file_handler)
    log.propagate = False
    return log


logger = _build_logger()
