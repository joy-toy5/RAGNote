# syntax=docker/dockerfile:1.7

# 构建阶段：按锁文件安装依赖并构建前端。
FROM node:20-alpine AS builder

WORKDIR /build

COPY package.json package-lock.json ./

RUN npm config set registry https://registry.npmmirror.com && \
    npm ci

COPY . .

# VITE_* 在构建时替换；运行时环境变量不会改变静态产物。
RUN npm run build

# 运行阶段：托管静态资源，并按 nginx.conf 转发后端请求。
FROM nginx:alpine AS runtime

COPY --from=builder /build/dist /usr/share/nginx/html

COPY --from=builder /build/nginx.conf /etc/nginx/nginx.conf
