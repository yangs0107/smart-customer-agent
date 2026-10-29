"""文档解析：把 txt / pdf 统一读成字符串。"""

from pathlib import Path

from pypdf import PdfReader


def read_file(file_path: str | Path) -> str:
    """读取知识库文档。

    - pdf：逐页抽取文本后拼接
    - txt：直接按 utf-8 读取，容错跳过无法解码的字节
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError(f"不支持的文件类型：{suffix}")
