# syntax=docker/dockerfile:1
# 「Agent 设计」问答 Agent —— 生产镜像
# 构建：docker build -t agent-design-qa:latest .
# 国内加速：docker build --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ -t agent-design-qa:latest .

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai

# 构建时可选 pip 镜像源（只影响构建，不影响运行时）
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

WORKDIR /app

# 依赖层单独缓存：只有 requirements.txt 变化才重装
COPY requirements.txt ./
RUN pip install -r requirements.txt

# 应用代码（不含 .venv / data / logs / PDF / 测试）
COPY app ./app
COPY web ./web
COPY main.py ./
COPY deploy/entrypoint.sh /usr/local/bin/entrypoint

RUN chmod +x /usr/local/bin/entrypoint \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/data /app/logs /app/docs \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status==200 else 1)"

ENTRYPOINT ["entrypoint"]
CMD ["serve"]
