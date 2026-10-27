#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提交前敏感信息扫描（API key / 公网 IP / 自有域名 / 默认口令 / 私钥 / 敏感文件）。

用法::

    python scripts/check_secrets.py --staged     # 扫描 git 暂存区（pre-commit 用）
    python scripts/check_secrets.py --all        # 扫描所有“将纳入版本管理”的文件
    python scripts/check_secrets.py --path FILE  # 扫描单个文件

退出码：0 = 通过；1 = 发现敏感信息；2 = 环境/用法错误。

行内豁免：在需要保留示例的那一行末尾加 ``secret-scan: allow``。
"""

from __future__ import annotations

import argparse
import ipaddress
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 允许出现在仓库里的公开服务域名（模型/包管理/文档），不视为敏感。
ALLOW_DOMAINS = {
    "example.com",
    "example.org",
    "example.net",
    "example.test",
    "localhost",
    "api.deepseek.com",
    "deepseek.com",
    "dashscope.aliyuncs.com",
    "aliyuncs.com",
    "pypi.org",
    "pypi.tuna.tsinghua.edu.cn",
    "mirrors.aliyun.com",
    "github.com",
    "raw.githubusercontent.com",
    "docker.io",
    "registry-1.docker.io",
    "hub.docker.com",
    "docs.docker.com",
    "nginx.org",
    "letsencrypt.org",
    "certbot.eff.org",
    "get.docker.com",
    "python.org",
    "docs.python.org",
    "pip.pypa.io",
    "fastapi.tiangolo.com",
    "milvus.io",
    "zilliz.com",
    "modelcontextprotocol.io",
    "openai.com",
    "w3.org",
}

# 出现即视为敏感的文件名（.env.example 是模板，例外）。
SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".crt", ".keystore", ".jks"}
SENSITIVE_NAMES = {".env", "id_rsa", "id_ed25519", "credentials", ".npmrc", ".pypirc"}

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "data",
    "logs",
    "dist",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
}
MAX_BYTES = 2_000_000
SKIP_SUFFIXES = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico",
    ".zip", ".gz", ".tar", ".jar", ".sqlite3", ".pyc", ".woff", ".woff2",
}

# ---- 规则（敏感关键字用拼接构造，避免扫描器命中自身源码） ----
RE_API_KEY = re.compile(
    r"(?<![A-Za-z0-9])"
    + "s" + "k-"                      # OpenAI 兼容 key 前缀
    + r"[A-Za-z0-9_-]{16,}"
)
RE_AWS_KEY = re.compile(r"(?<![A-Z0-9])" + "AK" + r"IA[0-9A-Z]{16}(?![A-Z0-9])")
RE_GITHUB_TOKEN = re.compile(r"(?<![A-Za-z0-9])" + "g" + r"hp_[A-Za-z0-9]{30,}")
RE_SLACK_TOKEN = re.compile(r"(?<![A-Za-z0-9])" + "x" + r"ox[baprs]-[A-Za-z0-9-]{10,}")
RE_PRIVATE_KEY = re.compile(r"-{5}BEGIN [A-Z ]{0,24}PRIVATE KEY-{5}")
RE_DEFAULT_CRED = re.compile(r"\b" + "r" + "oot:" + r"[^\s<]{4,}")
RE_CRED_ASSIGN = re.compile(
    r"(?i)\b(?:api[_-]?key|apikey|access[_-]?token|secret|passwd|password|token)\b"
    r"\s*[:=]\s*[\"']?"
    r"(?!<|\$\{|(?:settings|os|self|config|environ|getenv|app)\.[A-Za-z_])"
    r"[A-Za-z0-9_./+=-]{16,}"
)

RE_IPV4 = re.compile(
    r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"
)
RE_DOMAIN = re.compile(
    r"(?<![@\w.-])"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"(?:com|cn|net|org|io|dev|app|xyz|top|site|online|cloud|tech|ai)"
    r"(?![A-Za-z0-9_.-])",   # `args.top_k` 这类属性访问不算域名
    re.IGNORECASE,
)

# example.com / example.org / ... 的子域同样视为文档示例
EXAMPLE_BASES = ("example.com", "example.org", "example.net", "example.test")

DOC_NETS = [
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
]


def _is_public_ip(text: str) -> bool:
    """只拦截公网 IPv4：回环/私网/保留/文档地址不算敏感。"""
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return False
    if (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
        or ip.is_reserved
    ):
        return False
    if any(ip in net for net in DOC_NETS):
        return False
    return True


def _is_allowed_domain(domain: str) -> bool:
    """白名单服务域及其子域放行；其余视为自定义域名。"""
    if domain in ALLOW_DOMAINS:
        return True
    return any(domain.endswith("." + base) for base in ALLOW_DOMAINS)


def _mask(value: str) -> str:
    value = value.replace("\ufeff", "").strip()
    if len(value) <= 8:
        return value[:2] + "***"
    return value[:6] + "***" + value[-2:]


def _scan_line(line: str) -> list[str]:
    if "secret-scan: allow" in line:
        return []
    kinds: list[str] = []

    for rule in (RE_API_KEY, RE_AWS_KEY, RE_GITHUB_TOKEN, RE_SLACK_TOKEN):
        if rule.search(line):
            kinds.append("api-key")
            break
    if RE_PRIVATE_KEY.search(line):
        kinds.append("private-key")
    if RE_DEFAULT_CRED.search(line):
        kinds.append("default-credential")
    if RE_CRED_ASSIGN.search(line):
        kinds.append("credential-assignment")

    for match in RE_IPV4.finditer(line):
        if _is_public_ip(match.group(0)):
            kinds.append("public-ip")
            break

    for match in RE_DOMAIN.finditer(line):
        domain = match.group(0).lower()
        if not _is_allowed_domain(domain):
            kinds.append("custom-domain")
            break

    return kinds


def _sensitive_filename(path: str) -> bool:
    name = Path(path).name
    if name == ".env.example":
        return False
    if name in SENSITIVE_NAMES:
        return True
    if name.startswith(".env.") and name != ".env.example":
        return True
    return Path(name).suffix.lower() in SENSITIVE_SUFFIXES


def _scan_text(path: str, text: str) -> list[tuple[str, int, str, str]]:
    findings: list[tuple[str, int, str, str]] = []
    if _sensitive_filename(path):
        findings.append((path, 0, "sensitive-file", "文件名本身不应提交"))
    for lineno, line in enumerate(text.splitlines(), start=1):
        for kind in _scan_line(line):
            findings.append((path, lineno, kind, _mask(line.strip())))
    return findings


def _should_skip(path: str) -> bool:
    parts = Path(path).parts
    if any(part in SKIP_DIRS for part in parts[:-1]):
        return True
    return Path(path).suffix.lower() in SKIP_SUFFIXES


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout


def _files_from_git(staged: bool) -> list[str]:
    if staged:
        out = _git("diff", "--cached", "--name-only", "--diff-filter=ACM", "-z")
    else:
        out = _git("ls-files", "-z", "-c", "-o", "--exclude-standard")
    return [p for p in out.split("\0") if p]


def _read_staged(path: str) -> str | None:
    try:
        data = subprocess.run(
            ["git", "show", f":{path}"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return None
    if len(data) > MAX_BYTES:
        return None
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


def _read_worktree(path: str) -> str | None:
    full = ROOT / path
    try:
        if not full.is_file() or full.stat().st_size > MAX_BYTES:
            return None
        return full.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return None


def scan(staged: bool) -> list[tuple[str, int, str, str]]:
    findings: list[tuple[str, int, str, str]] = []
    for path in _files_from_git(staged):
        if _should_skip(path):
            continue
        text = _read_staged(path) if staged else _read_worktree(path)
        if text is None:
            findings.extend(
                _scan_text(path, "") if _sensitive_filename(path) else []
            )
            continue
        findings.extend(_scan_text(path, text))
    return findings


def scan_paths(paths: list[str]) -> list[tuple[str, int, str, str]]:
    findings: list[tuple[str, int, str, str]] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = (ROOT / raw).resolve()
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"[secret-scan] 无法读取 {raw}: {exc}", file=sys.stderr)
            continue
        findings.extend(_scan_text(str(path), text))
    return findings


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认 GBK：统一成 UTF-8 + 替换错误字符，避免扫描结果打印崩溃
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="提交前敏感信息扫描")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--staged", action="store_true", help="扫描 git 暂存区（默认）")
    group.add_argument("--all", action="store_true", help="扫描所有将纳入版本管理的文件")
    group.add_argument("--path", action="append", default=[], metavar="FILE", help="扫描指定文件")
    args = parser.parse_args(argv)

    try:
        if args.path:
            findings = scan_paths(args.path)
        else:
            findings = scan(args.staged or not args.all)
    except RuntimeError as exc:
        print(f"[secret-scan] 环境错误：{exc}", file=sys.stderr)
        return 2

    if not findings:
        print("[secret-scan] 通过：未发现敏感信息")
        return 0

    print("[secret-scan] 发现敏感信息，已阻止本次提交：\n")
    for path, lineno, kind, detail in findings:
        location = f"{path}:{lineno}" if lineno else path
        print(f"  {location}  [{kind}]  {detail}")
    print(
        "\n处理方式：\n"
        "  1. 删除/替换为占位符（如 <SERVER_IP>、<YOUR_DOMAIN>、<API_KEY>），或从 .env 读取；\n"
        "  2. 确认是安全的公开示例时，在该行末尾加注释 `secret-scan: allow`；\n"
        "  3. 若已提交过，需轮换凭据并清理 git 历史（见 docs/SECRETS.md）。"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
