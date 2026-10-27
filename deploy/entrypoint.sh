#!/usr/bin/env bash
# 容器入口：
#   entrypoint serve                 # 默认，启动 HTTP 服务
#   entrypoint check                 # 环境/连通性自检
#   entrypoint ingest [--book all]   # 离线入库（需要 PDF 与 data 挂载）
#   entrypoint stats | books
#   entrypoint ask "问题" [--book design]
set -euo pipefail

cmd="${1:-serve}"
if [ "$#" -gt 0 ]; then shift; fi

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

case "$cmd" in
  serve)
    exec python -m uvicorn app.main:app \
      --host "$HOST" \
      --port "$PORT" \
      --workers "${WEB_CONCURRENCY:-1}" \
      --proxy-headers \
      --forwarded-allow-ips='*' \
      --log-level "${LOG_LEVEL:-info}"
    ;;
  check|ingest|stats|books|ask|search)
    exec python -m app.cli "$cmd" "$@"
    ;;
  help|-h|--help)
    sed -n '2,9p' /usr/local/bin/entrypoint
    exit 0
    ;;
  *)
    echo "未知命令: $cmd（可用: serve|check|ingest|stats|books|ask|search）" >&2
    exit 64
    ;;
esac
