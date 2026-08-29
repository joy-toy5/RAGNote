"""RAG-019：确认 `ALIYUN_BASE_URL` 该指哪里，以及端点是否校验模型名。

为什么需要它：`factory.py` 的 ALIYUN 分支把 `base_url` 传给 `ChatTongyi`，而该类
无此字段且 `model_config` 为 `extra='ignore'`，参数被静默丢弃。修法有两条路，选哪条
取决于事实而不是偏好：

1. 若 `.env` 配的模型在**原生 DashScope 端点**可用，则不必换客户端，只需删掉那个
   无效参数（少一个依赖）。
2. 若只在**兼容模式端点**可用，则必须换成能真正接受 `base_url` 的客户端。

同时做一个对照：拿一个明显不存在的模型名打兼容模式端点。如果它也返回 200，说明该端点
不校验模型名，那么"兼容模式 200"就不能作为"模型存在"的证据——这个对照决定了上面的
结论是否可信。

只读探针，不改任何配置，不写产物。刻意不叫 `m3_*`：同 `source_fingerprint` 理由。

用法::

    PYTHONPATH=. .venv/bin/python scripts/rag019_endpoint_probe.py
"""

from __future__ import annotations

import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

NATIVE = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
COMPAT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
TIMEOUT = 30.0

# 对照用的假模型名：任何真实服务都不该认识它。
BOGUS_MODEL = "qwen-definitely-not-a-real-model-9999"


def probe_native(model: str, api_key: str) -> tuple[int, str]:
    """原生 DashScope 协议：模型名在顶层 `model`，输入在 `input.messages`。"""
    response = httpx.post(
        NATIVE,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "input": {"messages": [{"role": "user", "content": "ping"}]},
            "parameters": {"max_tokens": 4},
        },
        timeout=TIMEOUT,
    )
    return response.status_code, response.text[:220]


def probe_compat(model: str, api_key: str) -> tuple[int, str]:
    """OpenAI 兼容协议：`messages` 在顶层。"""
    response = httpx.post(
        COMPAT,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 4,
        },
        timeout=TIMEOUT,
    )
    return response.status_code, response.text[:220]


def main() -> int:
    api_key = os.getenv("ALIYUN_ACCESS_KEY_SECRET")
    if not api_key:
        print("缺少 ALIYUN_ACCESS_KEY_SECRET", file=sys.stderr)
        return 2
    print(f"密钥尾 6 位: …{api_key[-6:]}")

    env_model = os.getenv("ALIYUN_MODEL_NAME") or os.getenv("CHAT_MODEL_NAME") or "qwen3-max"
    print(f"ALIYUN_BASE_URL = {os.getenv('ALIYUN_BASE_URL')!r}")
    print(f"待测模型（来自 .env）= {env_model!r}\n")

    # (标签, 模型, 探针) —— qwen-max 是已知可用的正对照，BOGUS 是负对照。
    cases = [
        (f"原生端点   / {env_model}", env_model, probe_native),
        (f"兼容模式   / {env_model}", env_model, probe_compat),
        ("原生端点   / qwen-max（正对照）", "qwen-max", probe_native),
        ("兼容模式   / qwen-max（正对照）", "qwen-max", probe_compat),
        (f"兼容模式   / {BOGUS_MODEL}（负对照）", BOGUS_MODEL, probe_compat),
    ]

    for label, model, probe in cases:
        try:
            status, body = probe(model, api_key)
        except Exception as exc:  # noqa: BLE001 - 探针要把任何失败原样报出来
            print(f"{label:52s} -> 异常 {type(exc).__name__}: {exc}")
            continue
        verdict = "OK" if status == 200 else "拒绝"
        print(f"{label:52s} -> {status} {verdict}")
        if status != 200:
            print(f"{'':52s}    {body}")

    print(
        "\n判读：若负对照也返回 200，则兼容模式端点不校验模型名，"
        "\n      '兼容模式 200' 不能作为模型存在的证据，需另找依据。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
