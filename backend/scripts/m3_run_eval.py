"""在隔离评测索引上跑生产检索路径，产出冻结 run 产物。

本脚本只做召回基线，显式关掉两个不可复现的阶段，并把关闭这件事写进
retrieval_config，让 config_sha256 直接绑定这个决定：

1. HyDE：`generate_hypothetical_document` 每条 Query 调一次 LLM，输出不可复现，
   basline 不能建立在随机改写过的查询上。这里用原始 query 检索 —— 与生产在
   LLM 失败时的降级行为完全一致（rag_service 的 except 分支就是 return query）。
2. Reranker：Qwen3-Reranker-0.6B 的 checkpoint 是 CausalLM，没有 score 头。
   sentence-transformers 的 CrossEncoder 会随机初始化 `score.weight`，同一份文件
   连续加载两次打分不同，因此它的排序不是模型能力而是噪声，不能进基线。
   跳过时显式写 outcome="skipped" + error_code，报告里可审计。

用法::

    PYTHONPATH=. .venv/bin/python scripts/m3_run_eval.py \
        --dataset evals/datasets/m3_dev_v2/manifest.json \
        --index-dir evals/indexes/m3_dev_v2 \
        --output evals/runs/m3_dev_v2_retrieval_only.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

# LangSmith 上传会带来网络依赖和不确定性，离线 run 必须先关掉再导入任何 app 模块。
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from app.evaluation.contracts import ExecutorDescriptor, sha256_json  # noqa: E402
from app.evaluation.dataset import load_dataset  # noqa: E402
from app.evaluation.runner import EvalRequest, run_dataset, write_run  # noqa: E402
from app.rag.evaluation_adapter import RagServiceQueryExecutor  # noqa: E402
from app.rag.retrieval_contract import RetrievalTrace, StageObservation  # noqa: E402
from app.rag.retrievers.tokenization import TOKENIZER_ID  # noqa: E402
from app.utils.config import chroma_config  # noqa: E402

EXECUTOR_ID = "rag-note.offline-retrieval-only.v1"
COLLECTION_NAME = "m3-dev-v2-eval"
RERANK_SKIP_CODE = "RERANK_DISABLED_UNTRAINED_HEAD"
# 生产默认；从 chroma.yaml 读，避免脚本与生产各持一份权重。
DEFAULT_FUSION_WEIGHTS = (
    chroma_config["bm25"]["vector_weight"],
    chroma_config["bm25"]["bm25_weight"],
)


class _EmptyNotesStore:
    """笔记候选没有 provenance 元数据，会让整条 Query 被拒；离线评测必须返回空。"""

    @staticmethod
    def similarity_search(*args: Any, **kwargs: Any) -> list:
        return []


class _EmptyNoteService:
    def __init__(self) -> None:
        self.notes_store = _EmptyNotesStore()


class _RetrievalOnlyRagService:
    """包装生产 RagService，关掉 HyDE 与 reranker，并在出 trace 后立刻释放 client。"""

    def __init__(
        self,
        *,
        descriptor: ExecutorDescriptor,
        user_id: str,
        index_dir: str,
        top_k: int,
        retriever_mode: str,
        fusion_weights: tuple[float, float] | None = None,
    ) -> None:
        from app.rag.rag_service import RagService
        from app.rag.vector_store import VectorStoreService
        from app.utils.factory import embed_model

        self._descriptor = descriptor
        self._vector_store = VectorStoreService.for_explicit_target(
            persist_directory=index_dir,
            collection_name=COLLECTION_NAME,
            embedding_function=embed_model,
            top_k=top_k,
            fusion_weights=fusion_weights,
        )
        service = RagService(
            user_id=user_id,
            vector_store=self._vector_store,
            note_service_override=_EmptyNoteService(),
        )
        # 不改生产代码，只在评测实例上把两个不可复现阶段替换掉。
        service.generate_hypothetical_document = self._identity_hyde
        service.reorder_documents = self._skip_rerank
        if retriever_mode == "vector_only":
            # 消融用：预置检索器，让 initialize_retriever 的 None 判断直接短路，
            # 从而绕开 EnsembleRetriever，不改生产代码。
            service.retriever = self._vector_store.vectors_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k, "filter": {"user_id": user_id}},
            )
        elif retriever_mode != "hybrid":
            raise ValueError(f"不支持的 retriever_mode: {retriever_mode}")
        self._service = service
        self._closed = False

    @property
    def descriptor(self) -> ExecutorDescriptor:
        return self._descriptor

    @staticmethod
    async def _identity_hyde(query: str) -> str:
        return query

    @staticmethod
    async def _skip_rerank(query: str, candidates: list) -> list:
        return [
            candidate.observe(
                StageObservation(
                    stage="rerank",
                    route="cross_encoder",
                    rank=rank,
                    outcome="skipped",
                    error_code=RERANK_SKIP_CODE,
                )
            )
            for rank, candidate in enumerate(candidates, 1)
        ]

    async def get_retrieval_trace(self, query: str, *, query_id: str) -> RetrievalTrace:
        try:
            trace = await self._service.get_retrieval_trace(query, query_id=query_id)
        finally:
            self.close()
        if trace.candidates or trace.index_version is not None:
            return trace
        # 零候选拒答：index_version 来自候选元数据，没有候选就无从得知，生产
        # RagService 因此把它留成 None —— 它不该断言自己没观测到的版本。
        # 但 execution_from_trace 要求空候选 trace 也携带 index identity，否则
        # 一次拒答会以 ValueError 终止整条 run（_execute_query 对 ValueError
        # 原样上抛），拒答根本不会被计分。
        #
        # 由这里补齐是契约要求的形态：本工厂的 index_version 来自 manifest，而
        # manifest 已在开跑前与 dataset 比对、且与索引目录是同一次构建的产物，
        # 所以它是「由具体工厂验证的身份」，不是 request 的自报值。
        return replace(trace, index_version=self._descriptor.index_version)

    def close(self) -> None:
        """每次构造 Chroma 都让 SharedSystemClient refcount +2，只有 close 会减。"""
        if self._closed:
            return
        self._closed = True
        self._vector_store.close()


class OfflineRagServiceFactory:
    """逐 Query 返回独立服务实例；执行器会按对象身份拒绝复用。"""

    def __init__(
        self,
        *,
        index_version: str,
        retrieval_config: dict[str, Any],
        index_dir: str,
    ) -> None:
        mode = retrieval_config["retriever"]["mode"]
        self._descriptor = ExecutorDescriptor(
            executor_id=f"{EXECUTOR_ID}.{mode}",
            index_version=index_version,
            retrieval_config_sha256=sha256_json(retrieval_config),
            # 这条路只到检索为止：没有 LLM 调用，生成层拒答不可能被观测到。
            answer_path="retrieval_only",
        )
        self._index_dir = index_dir
        self._top_k = retrieval_config["top_k"]
        self._mode = mode
        # 权重从声明里回读，保证「声明的」与「执行的」是同一份值。
        declared = retrieval_config["retriever"].get("weights")
        self._fusion_weights = (
            (declared["vector"], declared["bm25"])
            if isinstance(declared, dict)
            else None
        )
        self._issued: list[_RetrievalOnlyRagService] = []

    @property
    def descriptor(self) -> ExecutorDescriptor:
        return self._descriptor

    def __call__(self, request: EvalRequest) -> _RetrievalOnlyRagService:
        service = _RetrievalOnlyRagService(
            descriptor=self._descriptor,
            user_id=request.user_id,
            index_dir=self._index_dir,
            top_k=self._top_k,
            retriever_mode=self._mode,
            fusion_weights=self._fusion_weights,
        )
        self._issued.append(service)
        return service

    def close_all(self) -> None:
        """兜底：任何没走到 get_retrieval_trace 的实例也必须释放。"""
        for service in self._issued:
            service.close()


def build_retrieval_config(
    top_k: int,
    mode: str = "hybrid",
    fusion_weights: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """所有影响检索结果的开关都必须在这里显式声明，config_sha256 才有意义。"""
    if mode == "hybrid":
        # top_k 是每路召回深度；EnsembleRetriever 做的是并集加权 RRF 融合，
        # 因此最终候选数可能大于 top_k。
        vector_weight, bm25_weight = fusion_weights
        retriever = {
            "mode": "hybrid",
            "type": "ensemble_rrf",
            "routes": ["chroma_similarity", "bm25"],
            "per_route_k": top_k,
            # 分词器身份必须进 config_sha256：换分词器就是换检索行为。
            "bm25_tokenizer": TOKENIZER_ID,
            # BM25 只返回打分为正的候选，并按 (score, 语料位置) 稳定排序；
            # 父类的 np.argsort 对 0 分文档一视同仁，会把零词法交集的文档
            # 按 numpy 分区次序送进融合（RAG-005 / RAG-015）。
            "bm25_zero_score_candidates": "dropped",
            "bm25_tie_order": "stable_by_corpus_position",
            "weights": {"vector": vector_weight, "bm25": bm25_weight},
            "weights_source": "m3_dev_v2_ablation_grid",
        }
    elif mode == "vector_only":
        retriever = {
            "mode": "vector_only",
            "type": "chroma_similarity",
            "routes": ["chroma_similarity"],
            "per_route_k": top_k,
            "weights": "n/a",
            "reason": "消融基线：量化 BM25 空白分词在中文查询上的代价",
        }
    else:
        raise ValueError(f"不支持的 retriever mode: {mode}")
    return {
        "contract": "rag-note.m3-retrieval-config.v1",
        "top_k": top_k,
        "retriever": retriever,
        "hyde": {
            "enabled": False,
            "reason": "LLM 改写不可复现，基线使用原始 query（等价于生产 LLM 失败降级）",
        },
        "reranker": {
            "enabled": False,
            "model": "Qwen/Qwen3-Reranker-0.6B",
            "reason": (
                "checkpoint 是 CausalLM，无 score 头；CrossEncoder 会随机初始化 "
                "score.weight，同一文件两次加载打分不同，排序是噪声而非模型能力"
            ),
        },
        "note_store": {
            "enabled": False,
            "reason": "笔记向量未版本化，无 provenance 元数据，不得进入正式 qrels",
        },
        "user_isolation": {"filter": "metadata.user_id", "enforced": True},
        "langsmith_tracing": False,
    }


def source_fingerprint(root: Path) -> str:
    """对影响检索行为的源码取稳定摘要，脏工作树也能被绑定。"""
    targets: list[Path] = []
    for pattern in ("app/**/*.py", "scripts/m3_*.py", "app/config/*.yaml"):
        targets.extend(
            path
            for path in root.glob(pattern)
            if "__pycache__" not in path.parts and path.is_file()
        )
    digest = hashlib.sha256()
    for path in sorted(set(targets), key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def source_revision(repo_root: Path) -> str:
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{head}-dirty" if dirty else head


async def run(arguments: argparse.Namespace) -> int:
    dataset = load_dataset(arguments.dataset)
    manifest = json.loads(Path(arguments.dataset).read_text(encoding="utf-8"))
    if manifest["index_version"] != dataset.index_version:
        raise SystemExit("manifest 与 dataset index_version 不一致")

    index_dir = Path(arguments.index_dir).resolve()
    if not (index_dir / "chroma.sqlite3").exists():
        raise SystemExit(f"{index_dir} 不像是已构建的 Chroma 索引")

    if arguments.retriever == "hybrid":
        weights = tuple(arguments.fusion_weights or DEFAULT_FUSION_WEIGHTS)
        if len(weights) != 2:
            raise SystemExit("--fusion-weights 需要两个值：向量权重 BM25权重")
    else:
        weights = None
    retrieval_config = build_retrieval_config(
        arguments.top_k, arguments.retriever, weights
    )
    backend_root = Path(__file__).resolve().parent.parent
    factory = OfflineRagServiceFactory(
        index_version=dataset.index_version,
        retrieval_config=retrieval_config,
        index_dir=str(index_dir),
    )
    executor = RagServiceQueryExecutor(factory)

    print(f"retriever      : {arguments.retriever}")
    print(f"queries        : {len(dataset.queries)}")
    print(f"index_version  : {dataset.index_version}")
    print(f"config_sha256  : {factory.descriptor.retrieval_config_sha256}")
    try:
        evaluation_run = await run_dataset(
            dataset,
            executor,
            source_revision=source_revision(backend_root.parent),
            source_fingerprint=source_fingerprint(backend_root),
            retrieval_config=retrieval_config,
        )
    finally:
        factory.close_all()

    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_run(evaluation_run, output)

    errored = [query for query in evaluation_run.queries if query.error_code]
    counts = sorted(len(query.candidates) for query in evaluation_run.queries)
    print(f"run_id         : {evaluation_run.run_id}")
    print(f"execution_sha  : {evaluation_run.execution_sha256}")
    print(f"candidates     : min={counts[0]} max={counts[-1]}")
    print(f"errored        : {len(errored)}")
    for query in errored:
        print(f"  {query.query_id}: {query.error_code}")
    print(f"写入 {output}")
    return 1 if errored else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="在隔离索引上产出 M3 run")
    parser.add_argument("--dataset", required=True, help="manifest.json 路径")
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--retriever",
        choices=("hybrid", "vector_only"),
        default="hybrid",
        help="hybrid 为生产路径；vector_only 为消融基线",
    )
    parser.add_argument(
        "--fusion-weights",
        type=float,
        nargs=2,
        metavar=("VECTOR", "BM25"),
        default=None,
        help=(
            "hybrid 融合权重，默认取 chroma.yaml。显式传入用于权重消融："
            "权重进 retrieval_config 从而进 config_sha256，源码指纹不变"
        ),
    )
    arguments = parser.parse_args()
    return asyncio.run(run(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
