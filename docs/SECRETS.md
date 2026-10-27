# 敏感信息与提交规范（SECRETS）

本仓库**禁止**出现任何真实的环境敏感数据。配置一律放在本地 `.env`（已被 `.gitignore` 忽略），
仓库里只保留带占位符的 `.env.example`。

## 一、禁止提交的内容

| 类别 | 示例/说明 |
|---|---|
| API Key / Token | 任意模型、Embedding、云服务的真实 key；私钥文件 |
| 服务器地址 | 公网 IP、内网 IP、`host:port` 形式的真实服务地址 |
| 域名 | 自己的根域名、子域名、证书与证书路径（证书文件本身也不提交） |
| 数据库凭据 | Milvus / 其它数据库的真实账号口令（含默认口令） |
| 个人信息 | 手机号、邮箱列表、身份证、客户数据 |
| 运行产物 | `.env`、`data/`、`logs/`、`dist/`、测试临时目录 |

允许出现：`127.0.0.1`、`0.0.0.0`、`localhost`、`example.com`、模型厂商公开 API 域名
（如 `api.deepseek.com`、`dashscope.aliyuncs.com`），以及 `<SERVER_IP>`、`<YOUR_DOMAIN>` 这类占位符。

## 二、三层防护

1. **`.gitignore`**：`.env`、`data/`、`logs/`、`dist/`、`*.pem`、`*.key`、`*.crt`、测试临时目录。
2. **提交前钩子**（pre-commit）：`scripts/check_secrets.py --staged`，命中即拒绝提交。
3. **CI**：`.github/workflows/secret-scan.yml` 在 push / PR 时执行 `--all` 扫描。

### 启用钩子（每个克隆做一次）

```bash
# Linux / macOS / Git Bash
sh scripts/install_git_hooks.sh

# Windows PowerShell
pwsh scripts/install_git_hooks.ps1
```

等价于：

```bash
git config core.hooksPath .githooks
```

### 手工自检

```bash
python scripts/check_secrets.py --all      # 扫所有将纳入版本管理的文件
python scripts/check_secrets.py --staged   # 只看暂存区
python scripts/check_secrets.py --path app/config.py
```

### 误报豁免

确实需要保留的公开示例（例如文档里的占位说明），在该行末尾加：

```text
# secret-scan: allow
```

豁免只对当前行生效，不要用它放过真实凭据。

## 三、配置写法

`.env.example` 只写占位符：

```dotenv
EMBEDDING_API_KEY=<DASHSCOPE_API_KEY>
LLM_API_KEY=<DEEPSEEK_API_KEY>
SYMBOL_AGENT_MILVUS_URI=http://<MILVUS_HOST>:19530
SYMBOL_AGENT_MILVUS_TOKEN=<MILVUS_TOKEN>
ADMIN_API_KEY=<RANDOM_32_BYTES>
```

本地 `.env`：

```bash
cp .env.example .env
chmod 600 .env
# 填入真实值；确保 `git status` 里看不到 .env
git check-ignore -v .env
```

> 永远不要 `git add -f .env`；不要在文档、测试、注释、截图里写真实值。

## 四、已经发生泄漏时怎么办

**顺序很重要：先轮换凭据，再清理历史。** 只清历史而不轮换，等于没修。

### 1. 立即轮换

- 修改数据库口令（Milvus 等），不要使用默认账号口令；
- 防火墙/安全组只放行应用服务器访问数据库端口，禁止公网直连；
- 轮换所有可能暴露的 API key（模型/Embedding/云厂商）；
- 检查数据库、对象存储、日志是否有异常访问；必要时从备份恢复。

### 2. 清理 Git 历史（二选一）

> 操作前备份：`cp -r .git ../repo-git-backup-$(date +%s)`。
> 强推会改写远端历史，需通知所有协作者重新克隆。

**方式 A：重建为单条干净历史（适合仓库刚起步，最彻底）**

```bash
git checkout --orphan clean-history
git add -A
git commit -m "chore: sanitize repository and add secret guard"

git branch -D master
git branch -m master
git push -f origin master

git reflog expire --expire=now --all
git gc --prune=now
```

**方式 B：保留历史但替换敏感文本（需要 `git-filter-repo`）**

```bash
pip install git-filter-repo

# 把真实旧值写在仓库外的临时文件里（不要提交这个文件）
# 每行格式：旧文本==>新文本
cat > /tmp/replacements.txt <<'EOF'
<OLD_IP>==><SERVER_IP>
<OLD_DOMAIN>==><YOUR_DOMAIN>
<OLD_DEFAULT_CRED>==><MILVUS_TOKEN>
EOF

git filter-repo --force --replace-text /tmp/replacements.txt \
  --invert-paths --path '*.pdf' --path '.env' --path '.claude' --path '.idea' --path 'pytest-of-*'

git remote add origin <REPO_URL>
git push -f origin master
```

注意：

- 远端旧 commit 的 SHA 可能在一段时间内仍可被直接访问，**轮换凭据才是根本**；
- 如果仓库是公开的，建议同时检查 GitHub 的 fork / 缓存 / 安全告警；
- 清理后让协作者重新 clone，不要继续用旧克隆推送。

### 3. 本项目的历史情况（重要）

公开仓库的既有提交中曾经包含过：**Milvus 地址与默认账号口令、自家域名与证书信息、服务器目录结构**。
API key 未进入 Git（`.env` 未被跟踪），但仍建议一并轮换。

处理建议：**先改 Milvus 口令并限制 19530 公网访问**，再按方式 A 重建历史并强推；
若该仓库已被他人 fork 或镜像，视为凭据已泄露处理。

## 五、额外建议

- **受版权保护的 PDF 原文不建议进 Git**：`git rm --cached *.pdf` 后提交（本地文件保留），
  入库需要的语料放在服务器或对象存储。历史中已存在的 PDF 需要按上面的方式清理。
- 审计日志 `logs/answers-*.jsonl` 含用户问题与引用原文，**不要提交**，按合规要求保留/脱敏。
- 二进制与构建产物（镜像 tar、`dist/`）不要进仓库；部署包通过制品库/对象存储分发。
- 截图与录屏在发布前检查是否包含地址栏域名、IP、密钥面板。

## 六、扫描规则摘要

`scripts/check_secrets.py` 会检查：

| 规则 | 说明 |
|---|---|
| `api-key` | 常见厂商 key 前缀（OpenAI 兼容、AWS、GitHub、Slack 等） |
| `private-key` | PEM/PGP 私钥块 |
| `default-credential` | 默认账号口令形态（如 `账号:口令` 且口令非占位符） |
| `credential-assignment` | `api_key=...` / `token: ...` 等 16 位以上真实赋值 |
| `public-ip` | 公网 IPv4（回环/私网/文档地址豁免） |
| `custom-domain` | 自有域名（公开服务/示例域名在白名单内） |
| `sensitive-file` | `.env`、`*.pem`、`*.key`、`*.p12` 等敏感文件被暂存 |

规则可通过修改 `scripts/check_secrets.py` 顶部的白名单调整。
