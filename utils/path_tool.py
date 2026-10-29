"""路径工具：把「相对项目根目录」的路径解析成绝对路径。

根目录基于 __file__ 推导，绝不用 os.getcwd()，
否则从 api/ 或 ui/ 启动时路径会随工作目录漂移。
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def get_abs_path(relative_path: str = "") -> str:
    return str((PROJECT_ROOT / relative_path).resolve())
