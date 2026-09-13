# syntax=docker/dockerfile:1.7

# 构建阶段：安装锁定依赖，编译工具不进入运行镜像。
FROM python:3.13-slim AS builder

# mysqlclient 源码构建需要 MySQL 开发库。
RUN sed -i 's@deb.debian.org@mirrors.tuna.tsinghua.edu.cn@g' \
        /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        pkg-config \
        default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        -i https://pypi.tuna.tsinghua.edu.cn/simple uv==0.11.7

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml uv.lock ./

# 仅安装第三方依赖，避免在源码复制前构建本地包。
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# 运行阶段：复用依赖环境。
FROM python:3.13-slim AS runtime

RUN sed -i 's@deb.debian.org@mirrors.tuna.tsinghua.edu.cn@g' \
        /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
        libmariadb3 \
        default-mysql-client \
    && rm -rf /var/lib/apt/lists/*

# 与后端统一 UID，便于共享上传卷。
RUN useradd --create-home --uid 1000 --shell /bin/bash app

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --chown=app:app . .

# 预建上传目录，并允许 Celery 在工作目录写日志。
RUN mkdir -p /app/media/img /app/logs && \
    chown -R app:app /app/media /app/logs && \
    chown app:app /app

USER app

EXPOSE 8001

# Celery 复用此镜像，由 Compose 覆盖启动命令。
CMD ["uvicorn", "DjangoUserService.asgi:application", "--host", "0.0.0.0", "--port", "8001"]
