"""探针：DashScope 兼容模式是否接受 `temperature=0`，以及设 0 后输出是否真的稳定。

为什么需要这个探针（`RAG-022`）：`RAG-019` 把 ALIYUN 对话客户端从 `ChatTongyi`
换成 `ChatOpenAI` 之后，`temperature` 从「客户端不暴露」变成「可传但未传」——
实测 `_default_params` 为 `{model, stream, top_p: 0.7}`，`temperature` 属性是
`None` 所以不进请求体。于是"生成不可复现"这条门禁第一次有了可操作的修法，但两件
事必须先实测，不能假定：

1. **兼容模式是否接受 `temperature=0`**。OpenAI 协议允许 0，但阿里云兼容层是
   自己实现的，是否接受、是否与 `top_p` 冲突都未验证过。
2. **接受不等于生效**。服务端可能接受参数却仍返回不同结果（例如后端做了
   batching 或 MoE 路由）。所以要重复调用比对逐字节相同率，而不是看它不报错。

配置对照，缺一不可：
- `A 当前生产`（无 temperature，`top_p=0.7`）——基线，证明现在确实会抖。
- `B temperature=0 + top_p=0.7`——最小改动方案（只加一个参数）。
- `C temperature=0 + top_p=1.0`——temperature=0 时 `top_p` 理论上已无作用，
  这一列用来验证"是否还需要一并改 top_p"，避免把两个改动混在一次断点里。
- `D temperature=0 + top_p=1.0 + seed`——第一轮实测 A/B/C 全部 4/4 每次都不同，
  说明 temperature 不是这里的确定性旋钮，所以补探 `seed`。
- `E 仅 seed`（保留生产 top_p，不设 temperature）——若 D 生效，这一列区分
  「是 seed 起作用」还是「必须 temperature+seed 同时设」，决定改动面有多大。

另记录每次调用的 `finish_reason` 与内容长度：第一轮 `qwen3.8-max` 配置 A 出现过
一次空内容（sha256 前 12 位 `e3b0c44298fc` 即空串），空回答会被评测判成拒答或错答，
必须查清是否可复现。

只读：不写任何产物，不改 `.env`，只打印密钥末 6 位。刻意不叫 `m3_*`——只读诊断
工具不该进 `source_fingerprint` 的 glob 而扰动后续每次 run。

用法：
    PYTHONPATH=. .venv/bin/python scripts/rag022_temperature_probe.py \
        [--repeat 4] [--configs A B] [--models qwen3.8-max]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.factory import build_aliyun_chat_model  # noqa: E402

# 固定 prompt。要求分点作答且留有措辞余地：完全封闭的问题即使在高 temperature 下
# 也可能字字相同，那样就无法证伪「设 0 之前会抖」这一半。
PROMPT = (
    "请用三句话说明：为什么在检索增强生成里，把检索到的文档数量从 3 篇提高到 8 篇，"
    "既可能提升答案完整性，也可能增加答错的风险。每句话独立成行。"
)

# (键, 标签, temperature, top_p, seed)
CONFIGS = [
    ("A", "当前生产（无 temperature, top_p=0.7）", None, 0.7, None),
    ("B", "temperature=0, top_p=0.7", 0.0, 0.7, None),
    ("C", "temperature=0, top_p=1.0", 0.0, 1.0, None),
    ("D", "temperature=0, top_p=1.0, seed=42", 0.0, 1.0, 42),
    ("E", "仅 seed=42（top_p=0.7, 无 temperature）", None, 0.7, 42),
]


def probe_one(
    model_name: str,
    temperature: float | None,
    top_p: float,
    seed: int | None,
    repeat: int,
):
    """返回 (状态, 请求体键, 每次调用的明细列表, 首次错误信息)。

    明细为 `(sha256 前 12 位, 内容长度, finish_reason)`，长度为 0 即空回答。
    """
    client = build_aliyun_chat_model(model_name=model_name, streaming=True, top_p=top_p)
    if temperature is not None:
        client.temperature = temperature
    if seed is not None:
        client.seed = seed
    body_keys = sorted(client._default_params)
    details: list[tuple[str, int, str]] = []
    for _ in range(repeat):
        try:
            message = client.invoke(PROMPT)
        except Exception as exc:  # noqa: BLE001 探针要把失败原样报出来
            return "REJECTED/ERROR", body_keys, details, f"{type(exc).__name__}: {exc}"
        text = message.content or ""
        finish = str(
            (message.response_metadata or {}).get("finish_reason")
            or (message.response_metadata or {}).get("stop_reason")
            or "?"
        )
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        details.append((digest, len(text), finish))
    return "ACCEPTED", body_keys, details, None


def drill_empty(model_name: str, repeat: int) -> None:
    """空回答专项钻取：区分「服务端没返回内容」与「客户端把内容丢了」。

    第一轮探针里 `qwen3.8-max` 出现过 `content == ""` 且 `finish_reason == "stop"`
    的回答。两种可能后果完全不同：

    - 若 `output_tokens > 0` 而 `content` 为空，说明模型确实生成了 token，但内容没
      落到 `content` 字段（例如落进了 `reasoning_content`），那是客户端/兼容层的
      解析问题，属生产 bug——线上用户也会拿到空回答。
    - 若 `output_tokens` 也为 0，说明服务端真的什么都没生成，属模型侧偶发。

    同时对 `streaming=True/False` 各跑一轮：生产对话路径是流式，评测脚本经由同一个
    工厂，若只有流式会空，那就是流式聚合的问题而非模型的问题。
    """
    print("===== 空回答专项钻取 =====")
    print(f"模型 {model_name}，每种 streaming 各 {repeat} 次，用生产参数（top_p=0.7）\n")
    for streaming in (True, False):
        client = build_aliyun_chat_model(
            model_name=model_name, streaming=streaming, top_p=0.7
        )
        if streaming:
            # 流式默认不返回 usage，必须显式开启才能拿到 output_tokens
            client.stream_usage = True
        empties = 0
        print(f"--- streaming={streaming} ---")
        for index in range(1, repeat + 1):
            try:
                message = client.invoke(PROMPT)
            except Exception as exc:  # noqa: BLE001 钻取要把失败原样报出来
                print(f"  #{index} 调用失败: {type(exc).__name__}: {exc}")
                continue
            text = message.content or ""
            usage = message.usage_metadata or {}
            extra = message.additional_kwargs or {}
            reasoning = extra.get("reasoning_content") or ""
            finish = str((message.response_metadata or {}).get("finish_reason") or "?")
            flag = ""
            if len(text) == 0:
                empties += 1
                flag = "  ← 空 content"
            print(
                f"  #{index} content_len={len(text):<5} "
                f"output_tokens={usage.get('output_tokens', '?'):<5} "
                f"reasoning_len={len(reasoning):<5} "
                f"finish={finish:<10} extra_keys={sorted(extra)}{flag}"
            )
        print(f"  空 content 次数: {empties} / {repeat}\n")


def drill_thinking(model_names: list[str], repeat: int) -> None:
    """思考 token 钻取：确认哪些模型在思考、思考能否关掉、关掉后是否变确定。

    钻取空回答时发现 `qwen3.8-max` 的 `completion_tokens=332` 里 `reasoning_tokens`
    占 249（75%），而 `content` 只有 134 字符、`reasoning_content` 为空——思考 token
    计费但被丢弃。这解释了两件事的可能成因，都要实测而非推断：

    - **为什么 temperature=0 与 seed 都不产生确定性**：若不可控的思考过程在前，
      后续答案自然跟着变。
    - **第 3 步成本估算**：按 `content` 长度估算会低估约 3 倍。

    `enable_thinking` 是 DashScope 的非 OpenAI 标准参数，只能走 `extra_body`。这里
    实测兼容层是否接受，以及关掉后 `reasoning_tokens` 是否真的归零、输出是否变确定。
    """
    print("===== 思考 token 钻取 =====")
    print(f"每模型每配置 {repeat} 次，生产参数 top_p=0.7、streaming=False\n")
    variants = [
        ("默认（不传 enable_thinking）", None),
        ("extra_body={'enable_thinking': False}", {"enable_thinking": False}),
    ]
    for model_name in model_names:
        print(f"--- {model_name} ---")
        for label, extra_body in variants:
            client = build_aliyun_chat_model(
                model_name=model_name, streaming=False, top_p=0.7
            )
            if extra_body is not None:
                client.extra_body = extra_body
            digests: list[str] = []
            reasoning_totals: list[int] = []
            failed: str | None = None
            for _ in range(repeat):
                try:
                    message = client.invoke(PROMPT)
                except Exception as exc:  # noqa: BLE001 钻取要把失败原样报出来
                    failed = f"{type(exc).__name__}: {exc}"
                    break
                text = message.content or ""
                details = (message.usage_metadata or {}).get("output_token_details", {})
                reasoning_totals.append(int(details.get("reasoning", 0)))
                digests.append(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12])
            print(f"  {label}")
            if failed:
                print(f"    调用失败   : {failed}")
                continue
            distinct = len(set(digests))
            print(f"    reasoning_tokens : {reasoning_totals}")
            print(f"    不同输出数       : {distinct} / {len(digests)}")
            print(
                "    判定             : "
                + ("确定性" if distinct == 1 else "非确定性")
                + f"；思考 token {'已归零' if not any(reasoning_totals) else '仍在产生'}"
            )
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="temperature 探针（只读，不落产物）")
    parser.add_argument("--repeat", type=int, default=4, help="每个配置重复调用次数")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen3.8-max", "qwen-max"],
        help="前者是生产 .env 配的模型，后者是 5.2.9/5.2.10 既有数字所用的模型",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[key for key, *_ in CONFIGS],
        help=f"只跑指定配置，可选 {[key for key, *_ in CONFIGS]}",
    )
    parser.add_argument(
        "--drill-empty",
        metavar="MODEL",
        help="只跑空回答专项钻取（区分服务端没返回 / 客户端丢内容），不跑配置对照",
    )
    parser.add_argument(
        "--drill-thinking",
        action="store_true",
        help="只跑思考 token 钻取（谁在思考、能否关掉、关掉后是否变确定）",
    )
    arguments = parser.parse_args()
    selected_configs = [c for c in CONFIGS if c[0] in arguments.configs]
    if not selected_configs:
        parser.error(f"--configs 未匹配到任何配置：{arguments.configs}")

    key = os.getenv("ALIYUN_ACCESS_KEY_SECRET") or ""
    base_url = os.getenv("ALIYUN_BASE_URL") or "(兜底常量)"
    print(f"密钥末 6 位 : ...{key[-6:] if key else '(未设置)'}")
    print(f"base_url    : {base_url}")
    print(f"每配置重复  : {arguments.repeat} 次")
    print(f"prompt 长度 : {len(PROMPT)} 字符\n")

    if arguments.drill_empty:
        drill_empty(arguments.drill_empty, arguments.repeat)
        return 0

    if arguments.drill_thinking:
        drill_thinking(arguments.models, arguments.repeat)
        return 0

    verdicts: dict[tuple[str, str], str] = {}
    for model_name in arguments.models:
        print(f"===== 模型 {model_name} =====")
        for key, label, temperature, top_p, seed in selected_configs:
            status, body_keys, details, error = probe_one(
                model_name, temperature, top_p, seed, arguments.repeat
            )
            digests = [d for d, _, _ in details]
            counts = collections.Counter(digests)
            distinct = len(counts)
            print(f"\n{key} {label}")
            print(f"  请求体键   : {body_keys}")
            print(f"  接受情况   : {status}")
            if error:
                print(f"  错误       : {error}")
                verdicts[(model_name, f"{key} {label}")] = "REJECTED"
                continue
            for index, (digest, length, finish) in enumerate(details, start=1):
                flag = "  ← 空回答" if length == 0 else ""
                print(
                    f"  #{index} sha={digest} len={length:<5} "
                    f"finish_reason={finish}{flag}"
                )
            empties = sum(1 for _, length, _ in details if length == 0)
            print(f"  不同输出数 : {distinct} / {len(digests)}")
            if empties:
                print(f"  空回答     : {empties} / {len(details)}")
            if distinct == 1:
                verdict = "确定性（逐字节相同）"
            elif distinct == len(digests):
                verdict = "每次都不同"
            else:
                verdict = f"部分相同（最高频出现 {counts.most_common(1)[0][1]} 次）"
            if empties:
                verdict += f"；含 {empties} 次空回答"
            print(f"  判定       : {verdict}")
            verdicts[(model_name, f"{key} {label}")] = verdict
        print()

    print("===== 汇总 =====")
    for (model_name, label), verdict in verdicts.items():
        print(f"{model_name:<14} {label:<44} {verdict}")
    print(
        "\n注意：'确定性' 只是本次 N 次调用内成立，不构成服务端保证。"
        "跨时间、跨服务端版本、跨 batching 状态都可能变。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
