#!/usr/bin/env bash
# 云端启动入口：平台通过 PORT 环境变量注入对外端口，必须绑定 0.0.0.0
set -e

PORT="${PORT:-3000}"

exec python3 -m uvicorn api.main:app --host 0.0.0.0 --port "$PORT"
