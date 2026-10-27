# 部署指南（DEPLOY）

面向把本项目发布到 Linux 服务器的完整步骤。两条路线：

| 路线 | 适用 | 优点 | 缺点 |
|---|---|---|---|
| **A. Docker Compose（推荐）** | 任意 Linux + Docker | 环境一致、依赖隔离、回滚简单 | 需要装 Docker，镜像约 860 MB |
| **B. systemd + venv** | 不能装 Docker / 已有 Python 运维体系 | 无容器开销 | 需要自己管 Python 依赖与权限 |

两种路线都遵循同一拓扑：

```
浏览器 ──HTTPS──> nginx(80/443) ──> uvicorn 127.0.0.1:8000
                                      ├── 阿里云百炼 Embedding（出网 443）
                                      ├── DeepSeek / Qwen LLM（出网 443）
                                      └── Milvus 2.6（出网 19530，现有实例）
```

> **不需要上传的东西**：两本 PDF（运行时不读；只有"在服务器上重新入库"才需要）、`.venv/`、`tests/`。
> **必须上传的**：`app/`、`web/`、`main.py`、`requirements.txt`、`deploy/`、`data/chunks.jsonl`、`.env`（服务器上现写）。

---

## 0. 服务器要求

| 项 | 最低 | 建议 |
|---|---|---|
| 系统 | Ubuntu 22.04 / 24.04、Debian 12、Rocky 9 | Ubuntu 24.04 |
| CPU / 内存 | 2 vCPU / 2 GB | 2 vCPU / 4 GB |
| 磁盘 | 5 GB | 20 GB（日志 + 版本镜像） |
| 出网 | 443 到 `dashscope.aliyuncs.com`、`api.deepseek.com`；19530 到 Milvus | 固定出口 IP，便于 Milvus 白名单 |
| 入网 | 80 / 443（有域名时） | 仅 80/443，**不要暴露 8000 和 Milvus** |

资源占用：应用常驻内存约 300–600 MB；CPU 几乎空闲（向量化与生成都在远端 API）。

---

## 1. 准备代码与目录

在**本机**项目根目录打包（排除虚拟环境、数据、PDF、缓存）：

```bash
# Linux/macOS
tar --exclude='.venv' --exclude='data' --exclude='logs' --exclude='.git' \
    --exclude='*.pdf' --exclude='__pycache__' --exclude='.pytest_cache' \
    -czf agent-design-book.tar.gz .

# Windows PowerShell
tar --exclude=.venv --exclude=data --exclude=logs --exclude=.git ^
    --exclude=*.pdf --exclude=__pycache__ -czf agent-design-book.tar.gz .
```

上传并解压到服务器：

```bash
sudo mkdir -p /opt/agent-design-book && sudo chown "$USER" /opt/agent-design-book
scp agent-design-book.tar.gz user@server:/tmp/
ssh user@server
tar -xzf /tmp/agent-design-book.tar.gz -C /opt/agent-design-book
cd /opt/agent-design-book
```

如果代码已在 Git 仓库：直接在服务器 `git clone` 到 `/opt/agent-design-book` 更省事（注意 `.env` 与 `data/` 不入库）。

---

## 2. 写生产配置 `.env`（两条路线通用）

```bash
cd /opt/agent-design-book
cp .env.example .env
chmod 600 .env
vi .env
```

生产必改项：

| 变量 | 生产建议 |
|---|---|
| `EMBEDDING_API_KEY` | 必填，DashScope Key；建议用独立子账号 Key |
| `EMBEDDING_MOCK` | **必须 0**（1 是本地桩向量，检索无意义） |
| `SYMBOL_AGENT_MILVUS_URI` / `SYMBOL_AGENT_MILVUS_TOKEN` | 指向生产 Milvus；**本项目实测应用与 Milvus 同机（<SERVER_IP>）→ 用 `http://127.0.0.1:19530`，并关闭公网 19530**；不要用公网 `<MILVUS_TOKEN>`，建最小权限账号 |
| `MILVUS_COLLECTION` | 保持 `agent_design_kb`（与入库时一致） |
| `ADMIN_API_KEY` | 必设（32 字节随机串）；否则 `/api/admin/*` 返回 503（默认关闭） |
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | DeepSeek 或百炼 Qwen；推理模型需 `LLM_MAX_TOKENS>=2048` |
| `RAG_TOP_K` | 默认 8，够用 |
| `TZ`（compose 注入） | `Asia/Shanghai` |

> 安全：`.env` 权限 600、属主为运行用户；**绝不进镜像、不进 Git**（`.gitignore` 已忽略）。

---

## 3. 准备 `data/chunks.jsonl`（关键）

`/api/stats`、`/api/books` 以及"重建索引/对照"都读这个文件；没有它可以问答，但统计与目录接口会为空。

**方式一：本机已入好库（推荐）**

Milvus 是**共享的远程实例**，本机入库后服务器直接连同一个 collection 即可，只需要把切片清单带过去：

```bash
# 本机
scp data/chunks.jsonl user@server:/opt/agent-design-book/data/
```

> 前提：本机与服务器 `.env` 指向**同一个 Milvus + 同一个 collection**。
> 若换 collection 或换 Milvus，必须在服务器重新入库（见方式二）。

**方式二：在服务器上入库**

1. 把两本 PDF 传到 `/opt/agent-design-book/`（Docker 部署需挂载，见下）。
2. 用容器或 venv 执行入库（会调用 Embedding API，全量约 1–3 分钟 + 少量费用）：

```bash
# Docker
docker compose run --rm app ingest --book all

# venv
.venv/bin/python -m app.cli ingest --book all
```

入库会写 `data/chunks.jsonl`、`data/embed_cache.sqlite3`，并生成 `docs/INGEST_REPORT.md`。
重复执行很快（缓存命中，约 15s）；`--recreate` 会重建 collection。

---

## 4. 路线 A：Docker Compose（推荐）

### 4.1 安装 Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER" && newgrp docker   # 免 sudo
docker compose version
```

### 4.2 构建并启动

```bash
cd /opt/agent-design-book
mkdir -p data logs

# 国内服务器建议走镜像源
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
  docker compose up -d --build
```

首次构建约 3–8 分钟（下载 Python 基础镜像 + 依赖）。查看状态：

```bash
docker compose ps
docker compose logs -f --tail=100 app
```

### 4.3 验证

```bash
# 容器内自检（Milvus 连通、collection、Embedding/LLM 配置、chunks）
docker compose run --rm app check

# 健康检查（宿主机）
curl -s http://127.0.0.1:8000/api/health | python3 -m json.tool

# 问一句
curl -s http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"提示链模式适合什么场景？"}' | python3 -m json.tool
```

> `check` 的判定口径：**Milvus 不可用 / collection 未创建 / Embedding 未配置**才算失败；
> `chunks.jsonl` 缺失、LLM 未配置、PDF 缺失只算警告（镜像默认不带 PDF，运行时不需要）。
> 如果你要在这台服务器上重新入库，请用 `check --require-pdfs` 把 PDF 缺失升级为错误。

`/api/health` 期望 `"status":"ok"`、`"milvus":true`、`"collection_ready":true`。
健康降级时看 `detail` 字段定位（Milvus 不通 / collection 不存在 / key 缺失）。

### 4.4 反向代理 + HTTPS

**方式 1：宿主机 nginx（常见）**

```bash
sudo apt install -y nginx
sudo cp deploy/nginx.conf /etc/nginx/conf.d/agent-design.conf
# 把 upstream 从 app:8000 改成 127.0.0.1:8000
sudo sed -i 's/server app:8000;/server 127.0.0.1:8000;/' /etc/nginx/conf.d/agent-design.conf
sudo nginx -t && sudo systemctl reload nginx
```

**方式 2：compose 内 nginx**

```bash
mkdir -p deploy/certs
docker compose -f docker-compose.yml -f docker-compose.nginx.yml up -d --build
```

然后申请证书（宿主机 nginx 示例）：

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d qa.example.com     # 按提示自动改写 443 配置
```

把 `deploy/nginx.conf` 中 80 端口的 `return 301` 与 443 server 块按需启用。

> **SSE 关键点**：`/api/chat/stream` 必须 `proxy_buffering off;` + `X-Accel-Buffering: no`，
> 否则前端只能一次性拿到全文、看不到流式输出。`nginx.conf` 已配好。

### 4.4.1 绑定已有域名（示例）

以一次实际环境探测为例（按自己的环境替换 `<...>`）：

| 项 | 现状 |
|---|---|
| `<YOUR_DOMAIN>` / `www.<YOUR_DOMAIN>` 解析 | 已指向 **<SERVER_IP>**（与 Milvus 同一台服务器） |
| 该服务器 80/443 | 已有 **nginx** 在跑（根域名当前返回 403，说明是默认站点/空站点） |
| 现有证书 | 已配置根域名/www 的 DV 证书（请注意到期时间，提前续期） |
| Milvus | 19530 端口在该机开放，且 `/api` 服务也在同机 |

由此得到两条结论：

1. **前端不需要单独部署**：FastAPI 在 `/` 直接托管 `web/`（`index.html`、`app.js`、`styles.css`），
   浏览器访问 `https://<YOUR_DOMAIN>/` 就是问答界面；前端用相对路径调 `/api/*`，同源，无需 CORS/环境变量。
2. **不要启动 compose 里的 nginx**（`docker-compose.nginx.yml`）：宿主机 80/443 已被占用，
   直接用宿主机 nginx 加一个 server 块即可。

#### 方案 1：直接用根域名（最快，复用现有证书）

DNS 已就绪，只改 nginx：

```bash
cd /opt/agent-design-book

# ① 确认没有同名 server 块（避免 conflicting server name，旧块优先）
sudo nginx -T | grep -n "server_name <YOUR_DOMAIN>"

# ② 查现有证书路径（腾讯云/宝塔/letsencrypt 各不相同）
sudo nginx -T | grep -nE "ssl_certificate|ssl_certificate_key"

# ③ 已备好配置：替换证书路径后安装
sudo cp deploy/nginx.domain.conf /etc/nginx/conf.d/agent-design.conf
sudo sed -i 's#/etc/nginx/ssl/<YOUR_DOMAIN>.pem#<实际证书路径>#' /etc/nginx/conf.d/agent-design.conf
sudo sed -i 's#/etc/nginx/ssl/<YOUR_DOMAIN>.key#<实际私钥路径>#' /etc/nginx/conf.d/agent-design.conf

# ④ 校验并热加载
sudo nginx -t && sudo systemctl reload nginx
```

访问：**https://<YOUR_DOMAIN>/**（HTTP 自动 301 到 HTTPS）。前端页面、`/api/health`、`/api/chat` 全部同域可用。

#### 方案 2：子域名隔离（推荐长期使用）

例如 `agent.<YOUR_DOMAIN>`，好处是根域名以后还能放别的站点：

```bash
# ① DNS：加一条 A 记录  agent -> <SERVER_IP>
# ② 确认解析（必须生效后再申请证书）
dig +short agent.<YOUR_DOMAIN>          # 期望输出 <SERVER_IP>
# ③ 申请证书（Let's Encrypt，自动续期）
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d agent.<YOUR_DOMAIN>
# ④ 把 deploy/nginx.domain.conf 的 server_name 改成 agent.<YOUR_DOMAIN>，
#    证书路径改成 /etc/letsencrypt/live/agent.<YOUR_DOMAIN>/...，再 reload
```

访问：**https://agent.<YOUR_DOMAIN>/**。根域名可以继续保留现状或 301 跳到该子域名。

#### 域名相关注意事项

- **现有证书请提前续期**：面板/云厂商签发的 DV 证书通常 3–12 个月到期，
  到期前在签发处续期，或改用 certbot + Let's Encrypt（自动续期，推荐）。续期后路径变化要同步 nginx。
- **如果服务器在中国大陆**：域名需完成 ICP 备案，否则 80/443 会被拦截。
  若 HTTPS 已能返回 403（而不是连接超时），一般说明备案/白名单已就绪，但建议再确认。
- **前端没有构建产物**：`web/` 是原生 HTML/JS/CSS，FastAPI 直接托管；改完前端刷新即可，无需发版。
- **支持子路径部署**：前端资源与 API 均为相对路径，可挂在 `/agent-book/` 下（见 4.4.3）。

### 4.4.2 堡垒面板 + 宿主机 nginx（项目目录 `/data/projects`）

实际环境示例（来自面板，按自己的环境替换）：

- nginx **由面板管理**，不在 Docker 里；
- `<YOUR_DOMAIN>` 的 443 站点已有自己的站点根目录与证书（相对文件名由面板生成）；
- 服务器上项目统一放在 `/data/projects`（已有其它项目目录）。

**目录约定**：本项目放 `/data/projects/agent-design-book`，与其它项目并列。

```bash
sudo mkdir -p /data/projects/agent-design-book
# 把代码上传/解压到该目录（或 git clone）
cd /data/projects/agent-design-book
cp .env.example .env && chmod 600 .env && vi .env
#   SYMBOL_AGENT_MILVUS_URI=http://127.0.0.1:19530    # 与 Milvus 同机，走回环
#   ADMIN_API_KEY=<32 字节随机串>
```

**运行应用（二选一）**

| 方式 | 命令 | 说明 |
|---|---|---|
| Docker（推荐） | `docker compose up -d --build` | 不污染宿主机 Python；端口已是 `127.0.0.1:8000` |
| venv + systemd | 见路线 B，路径改为 `/data/projects/agent-design-book` | 需要宿主机 Python ≥ 3.11（pymupdf 2.0 / pydantic 2.14 要求） |

**面板添加站点（推荐子域名 `agent.<YOUR_DOMAIN>`，不动现有站点）**

1. 面板 → 网站 → 添加站点：域名 `agent.<YOUR_DOMAIN>`，根目录
   `/data/projects/agent-design-book/web`，PHP 选“纯静态”。
2. 域名解析：加 A 记录 `agent` → `<SERVER_IP>`。
3. 站点 → SSL：一键申请 Let's Encrypt 证书（或上传已有证书）。
4. 站点 → 配置文件：粘贴 [deploy/nginx.panel-site.conf](nginx.panel-site.conf) 的内容
   （改 `server_name` 和证书文件名），保存（面板会自动 `nginx -t` 并 reload）。
5. 访问 **https://agent.<YOUR_DOMAIN>/**。

**如果一定要用根域名 `<YOUR_DOMAIN>`**

现有站点已有自己的根目录与其它 location。改前台等于替换现有首页，
需确认原首页可被替换；保留已有的其它 location，只做两件事：

```nginx
root /data/projects/agent-design-book/web;   # 改这一行
# 再加模板里的 /api/ 、/api/chat/stream 、/api/admin/ 三段 location
```

**最省事的变体（不改 root，整站反代）**：把现有 `location / { ... }` 换成

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_read_timeout 300s;
}
```

再单独加 `location = /api/chat/stream { ... proxy_buffering off; }`（SSE 必需）。
FastAPI 会同时托管前端与 API，功能一致。

**权限**

- nginx（面板通常是 `www` 用户）只需**读** `web/`，默认 755 即可。
- 应用需要**写** `data/`、`logs/`：
  - Docker：容器内是 uid 10001，若 bind mount 写不进 → `sudo chown -R 10001:10001 data logs`；
  - venv/systemd：`sudo chown -R <运行用户> data logs`。
- SELinux（Rocky/CentOS）：`sudo chcon -R -t httpd_sys_content_t /data/projects/agent-design-book/web`。

**验收**

```bash
nginx -t
curl -s https://agent.<YOUR_DOMAIN>/api/health | python3 -m json.tool   # status=ok, row_count=1289
curl -s https://agent.<YOUR_DOMAIN>/ | head -5                          # 返回 <title>Agent 设计问答…</title>
```

> 不要启动 `docker-compose.nginx.yml`：80/443 归面板 nginx，容器再占会端口冲突。
> 前端仍由 nginx 直接托管（模板已把 `root` 指向 `web/`），只把 `/api/` 反代到 8000。

### 4.4.3 挂到子路径 `/agent-book/`（不改现有首页）

需求：`https://<YOUR_DOMAIN>/agent-book/` 访问问答，而根域名现有站点与其它 location 保持不变。

前端已改为**相对路径**（`styles.css`、`app.js`、`api/chat`），因此放子路径不需要改代码，
由 nginx 做三件事：

1. `/agent-book/` → `alias` 到 `/data/projects/agent-design-book/web/`（静态页）；
2. `/agent-book/api/` → 剥离 `/agent-book` 前缀，反代到 `127.0.0.1:8000/api/`；
3. `/agent-book`（无尾斜杠）→ 301 到 `/agent-book/`，否则浏览器解析相对 API 路径会错位。

**步骤**

```bash
# 前提：应用已按 4.4.2 跑在 127.0.0.1:8000，且 /data/projects/agent-design-book 存在

# 面板 → 网站 → <YOUR_DOMAIN> → 配置文件
# 把 deploy/nginx.subpath-agent-book.conf 中「粘贴片段（start/end）」整段
# 加进现有 443 的 server { } 内（不要替换原有 root / location），保存。

# 命令行等价校验：
nginx -t
```

**访问**：`https://<YOUR_DOMAIN>/agent-book/`（输入不带斜杠会自动跳转）。

**已验证的链路**（本机用 nginx + 容器模拟相同配置）：

```
GET  /agent-book            -> 301 /agent-book/
GET  /agent-book/           -> 200（前端页面）
GET  /agent-book/app.js     -> 200
GET  /agent-book/api/health -> {"status":"ok","row_count":1289}
POST /agent-book/api/chat   -> route=deep 0.95，6 条引用，citation_ok=true
浏览器实际问答 + 引用面板渲染正常
```

**注意**

- `alias` 末尾的 `/` 不能省；`web/` 需 nginx（通常 `www`）可读（755）。
- 不需要 `try_files`：前端只有 3 个静态文件、没有前端路由；避免 `alias + try_files` 的已知坑。
- 如果现有站点里已有 `location ^~ /agent-book`、`rewrite` 或同名规则，先
  `nginx -T | grep -n agent-book` 检查，避免重复。
- 更新前端后浏览器强刷（静态资源未加长缓存）。
- 以后若想改回根域名/子域名，只需把 `alias` 换成 `root`（或整站反代），前端无需改动。

### 4.5 开机自启

`docker-compose.yml` 已设 `restart: unless-stopped`；确认 Docker 自身开机启动：

```bash
sudo systemctl enable docker
```

### 4.6 离线镜像交付（服务器无法访问 Docker Hub / PyPI）

生产服务器常常不能出网或拉取很慢，可以"本机构建 → 导出镜像 → 服务器导入"：

**本机（能出网、Docker 可用）**

```bash
docker build \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  -t agent-design-qa:latest .
docker save agent-design-qa:latest | gzip > agent-design-qa-latest.tar.gz   # 约 300 MB
scp agent-design-qa-latest.tar.gz user@server:/opt/agent-design-book/
```

**服务器**

```bash
cd /opt/agent-design-book
gunzip -c agent-design-qa-latest.tar.gz | docker load
docker compose up -d --no-build      # 直接使用导入的 agent-design-qa:latest
docker compose ps                    # 应为 healthy
```

> 服务器仍需能出网访问 **Embedding / LLM / Milvus**（它们不是容器内的服务）；
> 只有"构建镜像"这一步被离线化。


---

## 5. 路线 B：systemd + venv（无 Docker）

```bash
# 1) Python 3.12+（推荐 3.13；Ubuntu 24.04 自带 3.12 即可）
sudo apt install -y python3 python3-venv python3-pip nginx
cd /opt/agent-design-book
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt

# 2) 运行用户
sudo useradd --system --create-home --shell /usr/sbin/nologin agentqa
sudo chown -R agentqa:agentqa /opt/agent-design-book
sudo chmod 600 /opt/agent-design-book/.env

# 3) 自检（用运行用户执行，避免权限/环境差异）
sudo -u agentqa env $(grep -v '^#' .env | xargs) .venv/bin/python -m app.cli check

# 4) 安装 systemd 单元
sudo cp deploy/agent-design.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now agent-design
sudo systemctl status agent-design --no-pager
sudo journalctl -u agent-design -f

# 5) nginx 反代（同 4.4，upstream 用 127.0.0.1:8000）
```

升级：覆盖代码 → `.venv/bin/pip install -r requirements.txt` → `sudo systemctl restart agent-design`。

---

## 6. 运维手册

### 6.1 日志

| 日志 | 位置 | 说明 |
|---|---|---|
| 引用审计（需求 R3） | `logs/answers-YYYYMMDD.jsonl` | 每条回答的书/章/页/段/原文、路由、耗时 |
| 控制台 | `docker compose logs` / `journalctl -u agent-design` | 启动、路由判定、引用打印、报错 |
| 容器日志轮转 | compose `logging` 已限制 20 MB × 5 | 防磁盘打满 |

审计日志保留策略示例（保留 90 天，压缩归档）：

```bash
find /opt/agent-design-book/logs -name 'answers-*.jsonl' -mtime +90 \
  -exec gzip {} \; -exec mv {} /var/backups/agent-design/ \;
```

### 6.2 健康检查与监控

- 探活：`GET /api/health`，**必须判 `status=="ok"`**（HTTP 200 但 degraded 也算异常）。
- 建议接入：云监控 URL 探测（30s）、`/api/chat` 端到端延迟告警（>30s 告警）。
- 观察指标：问答耗时（`latency_ms`）、`citation_ok` 比例、`used_llm` 比例（降级异常时下降）。

### 6.3 备份

| 数据 | 说明 |
|---|---|
| `data/chunks.jsonl` | 切片清单（2.5 MB），丢失可重新入库重建 |
| `data/embed_cache.sqlite3` | 可选，丢失只是重新向量化（有费用） |
| `logs/` | 审计日志，按合规要求保留 |
| `.env` | 加密备份（含密钥） |
| Milvus `agent_design_kb` | 由 Milvus 侧备份；也可以 `ingest --recreate` 重建 |

```bash
tar -czf /var/backups/agent-design-$(date +%F).tar.gz \
  -C /opt/agent-design-book data logs .env
```

### 6.4 升级与回滚（Docker）

```bash
# 升级
cd /opt/agent-design-book && ./update.sh   # 或：git pull && docker compose up -d --build

# 回滚：保留上一个镜像 tag
APP_VERSION=v2 docker compose up -d
```

建议每次发布给镜像打版本 tag：`docker compose build -t agent-design-qa:v1.3 && docker compose up -d`。

### 6.5 重新入库（重建索引）

```bash
# 容器内（需要 PDF 挂载到 /app，见下方说明）
docker compose run --rm app ingest --book all --recreate

# 或走受保护的管理接口
curl -X POST http://127.0.0.1:8000/api/admin/ingest \
  -H "Authorization: Bearer $ADMIN_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"recreate_collection": true}'
```

> 容器默认**不打进 PDF**（`.dockerignore`）。要在容器内入库，先把两本 PDF 放到挂载目录：
> ```yaml
> volumes:
>   - ./data:/app/data
>   - ./Agent设计模式-zh-CN.pdf:/app/Agent设计模式-zh-CN.pdf:ro
>   - ./深入理解 AI Agent-zh-CN.pdf:/app/深入理解 AI Agent-zh-CN.pdf:ro
> ```

---

## 7. 安全清单（上线前逐项确认）

- [ ] 公网只开 22/80/443；`8000` 仅 `127.0.0.1`（compose 已默认）；**Milvus 19530 不对公网开放**
- [ ] Milvus 改掉 `<MILVUS_TOKEN>`，用最小权限账号 + IP 白名单
- [ ] `.env` 权限 600，属主为运行用户；未提交到 Git
- [ ] `ADMIN_API_KEY` 已设置且足够随机；nginx 层默认 `deny /api/admin/`
- [ ] HTTPS 已启用（certbot），HTTP 301 跳转
- [ ] 容器以非 root（uid 10001）运行；systemd 带 `NoNewPrivileges` / `ProtectSystem`
- [ ] 为 `/api/chat` 配置限流，防止模型费用被刷（见 7.1）
- [ ] 审计日志盘空间与保留策略已配置

### 7.1 可选：按 IP 限流（防刷）

```nginx
# http {} 块内
limit_req_zone $binary_remote_addr zone=chat:10m rate=10r/m;

# location /api/chat 内
limit_req zone=chat burst=5 nodelay;
limit_req_status 429;
```

---

## 8. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `/api/chat` 返回 **503** | 未入库 / Milvus 不通 / Embedding 配置错。先 `docker compose run --rm app check` 看结论 |
| `/api/health` 200 但 `degraded` | `detail` 里有原因：`collection_error`（collection 未建）或 `milvus_error`（网络/白名单） |
| `/api/stats` 行数 0、`/api/books` 为空 | `data/chunks.jsonl` 没挂载或没上传 |
| 前端不是流式输出 | nginx 未关闭缓冲：确认 `/api/chat/stream` 的 `proxy_buffering off` 与 `X-Accel-Buffering no` |
| 问答很慢（10–20s） | `deepseek-flash` 是推理模型，正常；换 `deepseek-chat` 可降到 3–5s |
| 容器里访问不到 Milvus | 服务器出网/安全组未放行 19530；或在 Milvus 侧加服务器出口 IP 白名单 |
| 容器内要访问宿主机代理 | compose 里加 `extra_hosts: ["host.docker.internal:host-gateway"]`，代理地址写 `host.docker.internal:7890` |
| 中文日志乱码 | 确认 `TZ=Asia/Shanghai`、`PYTHONUNBUFFERED=1`（镜像已设） |
| 端口被占用 | 改 `ports: "127.0.0.1:8001:8000"`，同步改 nginx upstream |
| 构建卡在 pip | 用 `PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/` 重新构建 |

---

## 9. 上线检查清单

1. `docker compose up -d --build` 后 `docker compose ps` 为 `healthy`
2. `docker compose run --rm app check` → 全部通过
3. `curl /api/health` → `status=ok`、`collection_ready=true`、`row_count=1289`
4. `curl /api/api/chat`（或前端）→ 有答案、有 `[n]` 引用、`citation_ok=true`
5. `logs/answers-*.jsonl` 出现记录，且含"书/章/页/段/原文"
6. 域名 HTTPS 可访问；`/api/admin/` 从公网 403/404
7. 重启服务器后服务自动恢复（`restart: unless-stopped` 或 systemd enabled）

---

## 10. 附：可选自建 Milvus

当前项目使用**现有远程 Milvus**，无需部署。若要把 Milvus 一起搬到本机服务器：

- 官方 standalone 依赖 etcd + MinIO，至少 2 GB 内存；
- 版本需与客户端 `pymilvus 3.0.2` 兼容（本项目实测服务端 2.6.0-rc.1）；
- 部署后用 `python -m app.cli ingest --recreate` 重新建 collection 并写入 1289 条向量。

生产更推荐使用云上托管 Milvus（Zilliz Cloud 等）或独立内网实例，避免与应用抢内存。
