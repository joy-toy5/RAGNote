# syntax=docker/dockerfile:1.7

# 构建阶段：安装锁定依赖，编译工具不进入运行镜像。
FROM python:3.12-slim AS builder

RUN sed -i 's@deb.debian.org@mirrors.tuna.tsinghua.edu.cn@g' \
        /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        -i https://pypi.tuna.tsinghua.edu.cn/simple uv==0.11.7

# 依赖环境与源码分离，避免挂载源码时覆盖依赖。
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml uv.lock ./

# 保留开发依赖，供镜像内测试使用。
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra dev

# 运行阶段：复用依赖环境。
FROM python:3.12-slim AS runtime

# libmagic1 用于文件类型识别，libgomp1 提供 OpenMP 运行库。
RUN sed -i 's@deb.debian.org@mirrors.tuna.tsinghua.edu.cn@g' \
        /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
        libmagic1 \
        libgomp1 \
        default-mysql-client \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 --shell /bin/bash app

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 保持 /app/app 布局，确保项目根目录与数据路径解析正确。
COPY --chown=app:app . .

RUN mkdir -p /app/data/chromadb /app/data/blobs /app/data/md5_hex_store \
             /app/data/extracted_images /app/logs && \
    chown -R app:app /app/data /app/logs

USER app

EXPOSE 8000

# 迁移服务复用此镜像，由 Compose 覆盖启动命令。
# run.py 处理 SIGTERM，停止接单后有界等待与清理。
CMD ["python", "run.py"]
