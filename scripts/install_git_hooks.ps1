# 启用仓库内提交钩子（只改本机 git 配置）
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
git config core.hooksPath .githooks
Write-Host "[secret-scan] 已启用提交前敏感信息扫描：core.hooksPath=.githooks"
Write-Host "[secret-scan] 手工检查：python scripts/check_secrets.py --all"
