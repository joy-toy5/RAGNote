-- 仅在 MySQL 首次初始化空数据卷时执行；已有数据卷不会重跑。

CREATE DATABASE IF NOT EXISTS `chat_history`
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE DATABASE IF NOT EXISTS `user_service`
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- mysql_native_password 用于兼容当前客户端；升级 MySQL 时需复核。
-- Demo 密码须与 Compose 一致，实际部署请替换。
CREATE USER IF NOT EXISTS 'rag'@'%'
  IDENTIFIED WITH mysql_native_password BY 'rag_app_password';
GRANT ALL PRIVILEGES ON `chat_history`.* TO 'rag'@'%';
GRANT ALL PRIVILEGES ON `user_service`.* TO 'rag'@'%';
FLUSH PRIVILEGES;
