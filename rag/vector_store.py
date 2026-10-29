"""知识库构建：扫描 data/ → MD5 去重 → 切分 → 写入 Chroma。

重复运行会跳过已入库文件（依据 md5.txt）。
注意：若只删 chroma_db/ 而保留 md5.txt，会导致全部文件被跳过却查不到内容，
重建知识库时两者要一起删。
"""

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embed_model
from utils.config_handler import chroma_conf
from utils.file_handler import read_file
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


def get_vector_store() -> Chroma:
    """统一走 langchain_chroma.Chroma，不直连 chromadb.PersistentClient。"""
    return Chroma(
        collection_name=chroma_conf["collection_name"],
        embedding_function=embed_model,
        persist_directory=get_abs_path(chroma_conf["persist_directory"]),
    )


def _load_md5() -> set[str]:
    path = Path(get_abs_path(chroma_conf["md5_hex_store"]))
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _save_md5(known: set[str]) -> None:
    Path(get_abs_path(chroma_conf["md5_hex_store"])).write_text(
        "\n".join(sorted(known)), encoding="utf-8"
    )


def _file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_store() -> None:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chroma_conf["chunk_size"],
        chunk_overlap=chroma_conf["chunk_overlap"],
        separators=chroma_conf["separators"],
    )
    store = get_vector_store()
    known = _load_md5()
    allowed = {s.lower() for s in chroma_conf["allow_knowledge_file_type"]}
    data_dir = Path(get_abs_path(chroma_conf["data_path"]))

    scanned = skipped = added = 0
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue

        scanned += 1
        file_md5 = _file_md5(path)
        if file_md5 in known:
            skipped += 1
            logger.info("跳过（已入库）| %s", path.name)
            continue

        text = read_file(path)
        if not text.strip():
            logger.warning("跳过（无文本）| %s", path.name)
            continue

        docs = [
            Document(page_content=chunk, metadata={"source": path.name, "file_md5": file_md5})
            for chunk in splitter.split_text(text)
            if chunk.strip()
        ]
        store.add_documents(docs)
        known.add(file_md5)
        added += len(docs)
        logger.info("入库 | %s → %d 个切片", path.name, len(docs))

    _save_md5(known)
    logger.info("建库完成 | 扫描 %d，跳过 %d，新增 %d 个切片，库内共 %d",
                scanned, skipped, added, store._collection.count())


if __name__ == "__main__":
    build_store()
