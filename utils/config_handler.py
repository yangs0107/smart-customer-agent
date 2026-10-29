"""配置加载：模块导入时一次性把 YAML 读进内存。

其他模块直接 from utils.config_handler import rag_conf 即可。
"""

from typing import Any

import yaml

from utils.path_tool import get_abs_path


def _load_yaml(filename: str) -> dict[str, Any]:
    with open(get_abs_path(f"config/{filename}"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


rag_conf: dict[str, Any] = _load_yaml("rag.yml")
chroma_conf: dict[str, Any] = _load_yaml("chroma.yml")
agent_conf: dict[str, Any] = _load_yaml("agent.yml")
prompts_conf: dict[str, Any] = _load_yaml("prompts.yml")
