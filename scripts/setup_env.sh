#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# 获取脚本所在目录和项目根目录

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found, please install uv first."
  exit 1
fi

cd "${ROOT_DIR}"

# 创建虚拟环境并安装依赖
uv venv
# 同步项目依赖
uv sync
uv pip install python-dotenv
