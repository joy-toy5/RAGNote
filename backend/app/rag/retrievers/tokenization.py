"""BM25 中文分词：CJK 字符 bigram（RAG-005）。

为什么不能用 langchain 的 default_preprocessing_func（`text.split()`）：
中文查询没有空格，整句退化成单个 token，而该 token 是一整句话，语料里永不出现。
在 m3_dev_v2 上实测，50 条查询有 23 条退化成 1 个 token，40 条可回答查询里 19 条
BM25 打分全零。此时 `BM25Retriever` 仍返回 20 条候选（`np.argsort` 对全零数组的
分区次序），并在融合里拿到权重 —— 故障不是「召回不准」，而是「把噪声当排名」。

为什么是 bigram 而不是单字：bigram 保留字序（`缓存` != `存缓`），本语料词表
1447 vs 512，区分度更高；这也是 Elasticsearch `CJKBigramFilter` 的做法。实测
`Recall@3` 0.9000（单字 0.8875）。

为什么不引入 jieba：bigram 零新依赖已达标（`Recall@3` 0.8625 -> 0.9000）。
jieba 0.42.1 最后发布于 2020-01，仅 sdist 18.3MB，且切词结果依赖内置词典与 HMM
模型 —— 要让 `config_sha256` 诚实，还得把词典版本一起绑进 attestation。

分词器身份必须写进 `retrieval_config`，否则 run 的 attestation 是假的。
"""

from __future__ import annotations

import re

# 分词器身份。改变分词行为必须同时改这个 id，让 config_sha256 跟着变。
TOKENIZER_ID = "cjk_bigram.v1"

# CJK 统一表意文字、扩展 A、兼容表意文字。逐字符成单元，拉丁/数字整词成单元。
_CJK_RANGES = "一-鿿㐀-䶿豈-﫿"
_UNIT = re.compile(rf"[{_CJK_RANGES}]|[0-9a-zA-Z_]+")
_IS_CJK = re.compile(rf"[{_CJK_RANGES}]")


def _pair_up(run: list[str]) -> list[str]:
    """连续 CJK 单字两两成词；单字成行时保留该字，避免丢词。"""
    if len(run) < 2:
        return list(run)
    return [run[i] + run[i + 1] for i in range(len(run) - 1)]


def cjk_bigram_tokenize(text: str) -> list[str]:
    """把文本切成 BM25 token：连续汉字取 bigram，拉丁/数字取整词，丢标点。

    查询侧与语料侧必须用同一个函数，否则两边词形不一致，匹配恒为空。

    :param text: 待分词文本；非字符串或空串返回空列表。
    :return: token 列表；切不出任何 token 时为空列表（调用方需据此跳过 BM25）。
    """
    if not isinstance(text, str) or not text:
        return []

    tokens: list[str] = []
    cjk_run: list[str] = []
    for unit in _UNIT.findall(text.lower()):
        if _IS_CJK.match(unit):
            cjk_run.append(unit)
            continue
        if cjk_run:
            tokens.extend(_pair_up(cjk_run))
            cjk_run = []
        tokens.append(unit)
    if cjk_run:
        tokens.extend(_pair_up(cjk_run))
    return tokens
