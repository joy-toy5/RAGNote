import json
import os
from typing import Any

import redis.asyncio as redis

# 从环境变量读取,默认值保持原有字面量 —— 宿主机裸跑的行为不变。
# 改动原因:容器内 localhost 指向容器自身,连不上独立的 redis 服务;
# 而 .env 里本就有 REDIS_HOST/REDIS_PORT/REDIS_DB 三个键,此前从未被读取。
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
# 注意默认值是 3,不是 .env.example 里写的 0 —— 与原代码保持一致,
# 避免改动连带切换了缓存所在的逻辑库。
REDIS_DB = int(os.getenv("REDIS_DB", "3"))

# 全局redis客户端对象
redis_client = None

async def connect_redis():
    """连接Redis"""
    global redis_client
    if redis_client is None:
        redis_client = redis.Redis(
            host=REDIS_HOST, # redis主机地址
            port=REDIS_PORT, # redis端口号
            db=REDIS_DB,     # redis数据库编号(0-15)
            decode_responses=True # 是否对返回值进行解码(True:返回字符串,False:返回字节)
        )
    return redis_client

async def close_redis():
    """关闭Redis连接"""
    global redis_client
    if redis_client:
        await redis_client.aclose()
        redis_client = None

async def check_redis_connection() -> bool:
    """检查Redis连接"""
    try:
        redis_client = await connect_redis()
        await redis_client.ping()
        return True
    except Exception as e:
        print(f"Redis连接失败: {e}")
        return False

# 设置和读取redis
async def get_redis_cache_str(key: str) -> str | None:
    """根据key获取redis缓存 (字符串类型)"""
    try:
        redis_client = await connect_redis()
        return await redis_client.get(key)
    except Exception as e:
        print(f"获取redis缓存失败: {e}")
        return None

async def get_redis_cache_json(key: str) -> dict | None:
    """根据key获取redis缓存 (字典或列表类型)"""
    try:
        redis_client = await connect_redis()
        data = await redis_client.get(key)
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        print(f"获取redis的JSON缓存失败: {e}")
        return None

async def set_redis_cache(key: str, value: Any, expire: int = 3600) -> bool:
    """
    根据key设置redis缓存

    :param key: 缓存键
    :param value: 缓存值
    :param expire: 过期时间(秒)
    :return: None
    """
    try:
        redis_client = await connect_redis()
        if isinstance(value, str):
            # 如果是字符串，直接设置缓存
            await redis_client.set(key, value, ex=expire)
        elif isinstance(value, (dict, list)):
            # 如果是字典或列表，转为json字符串在设置缓存
            await redis_client.set(key, json.dumps(value, ensure_ascii=False), ex=expire)
        else:
            # 其他类型，尝试转换为字符串
            await redis_client.set(key, str(value), ex=expire)
        return True

    except Exception as e:
        print(f"设置redis缓存失败: {e}")
        return False