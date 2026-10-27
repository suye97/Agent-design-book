#!/bin/sh
# 启用仓库内提交钩子（只改本机 git 配置，不影响其他人）
set -e
cd "$(dirname "$0")/.."
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit 2>/dev/null || true
echo "[secret-scan] 已启用提交前敏感信息扫描：core.hooksPath=.githooks"
echo "[secret-scan] 手工检查：python scripts/check_secrets.py --all"
