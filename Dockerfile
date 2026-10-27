FROM python:3.12-slim

# 离线优先 + 不写字节码 + 日志实时刷新
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HOST=0.0.0.0 \
    PORT=8001

WORKDIR /app

# curl 用于 HEALTHCHECK
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖，利用镜像层缓存
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

EXPOSE 8001

# 健康检查：/ok 就绪即视为健康
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8001/ok || exit 1

# 密钥通过 --env-file .env 注入，不打进镜像
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8001"]
