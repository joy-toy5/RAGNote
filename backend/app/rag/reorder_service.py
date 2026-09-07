from typing import List, Dict, Any
import torch
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import snapshot_download
from tqdm import tqdm
from app.core.logger_handler import logger
from app.rag.retrieval_contract import (
    RetrievalCandidate,
    StageObservation,
    validate_rerank_scores,
)

# 加载环境变量
load_dotenv()


def find_model_path(base_path: str) -> str:
    if os.path.exists(os.path.join(base_path, 'config.json')):
        return base_path
    
    for root, dirs, files in os.walk(base_path):
        if 'config.json' in files:
            return root
    
    logger.info(f"✅ 模型路径：{base_path}")
    logger.info(f"✅ 模型路径：{root}")
    return base_path


def check_and_download_reranker_model() -> None:
    """检查并重排序模型，在FastAPI启动时执行"""
    LOCAL_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", r"D:\Hugging_Face\models\Qwen3-Reranker-0.6B")
    MODELSCOPE_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"

    try:
        if os.path.exists(LOCAL_MODEL_PATH) and os.path.isdir(LOCAL_MODEL_PATH):
            logger.info(f"✅ 检测到本地重排序模型：{LOCAL_MODEL_PATH}")
        else:
            logger.warning(f"⚠️  本地模型未找到：{LOCAL_MODEL_PATH}")
            logger.info(f"🔄 开始从魔搭社区下载模型：{MODELSCOPE_MODEL_NAME}")

            os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

            with tqdm(total=100, desc='下载模型', leave=True, bar_format='{l_bar}{bar}| {n_fmt}%') as pbar:
                pbar.update(10)
                snapshot_download(
                    model_id=MODELSCOPE_MODEL_NAME,
                    cache_dir=LOCAL_MODEL_PATH,
                    revision='master'
                )
                pbar.update(90)

            logger.info(f"✅ 模型下载完成，保存路径：{LOCAL_MODEL_PATH}")

    except Exception as e:
        logger.error(f"❌ 模型检查失败: {str(e)}")
        raise RuntimeError(f"重排序模型检查失败: {str(e)}")


# ── Qwen3-Reranker 官方打分口径 ────────────────────────────────────────────────
# 该系列是**生成式判别器**，不是序列分类模型：checkpoint 里没有标量打分头
# （310 个张量中无 score/classifier/pooler），config 的 architectures 是
# ['Qwen3ForCausalLM']。用 CrossEncoder / AutoModelForSequenceClassification 加载
# 会让 transformers 随机初始化一个 score.weight 并**正常返回**——不抛异常，
# 打分是噪声，每次进程重启排序都不同。参见 `RAG-013`。
#
# 正确口径：按下面的模板拼 prompt，取最后一个位置的 logits，只在 yes / no 两个
# token 上做 log_softmax，取 yes 的概率作为相关性分数（0~1，higher_is_better）。
# 不能在全词表（151669）上归一化——那会把分数压到极小且互不可比。
#
# 模板必须与模型卡逐字一致：分数就是「assistant 第一个待生成位置」的分布，
# 改一个字符就换了一个测量口径。末尾那个空的 <think>\n\n</think> 不能省。
_RERANK_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on "
    'the Query and the Instruct provided. Note that the answer can only be "yes" or '
    '"no".<|im_end|>\n<|im_start|>user\n'
)
_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_RERANK_TASK = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
_RERANK_MAX_LENGTH = 2048

# 加载自检用的样本。刻意选得毫不相干，使判据不依赖模型的精细分辨力。
# 官方口径实测：相关 ~0.997，不相关 ~0.00005。阈值 0.5 只用来区分「有信号 / 无信号」。
_SELFTEST_QUERY = "如何重置密码"
_SELFTEST_RELEVANT = "在设置页面点击“重置密码”，输入原密码后提交。"
_SELFTEST_IRRELEVANT = "本产品的年度授权费用为每席位 1200 元。"
_SELFTEST_MIN_MARGIN = 0.5

# 批不变性自检用的长文档：刻意比上面那条短文档长得多，制造大量填充。
# 单对方向断言抓不住右填充（实测右填充下方向仍对、间隔 0.9700），
# 右填充真正破坏的是批不变性——同一文档的分数不得取决于同批还有谁。
_SELFTEST_LONG_FILLER = (
    "服务器机房位于华东二区，配备双路供电与柴油发电机组。"
    "机房温度维持在 22 摄氏度，湿度 45% 到 55% 之间。"
    "所有机柜配备独立配电单元，并接入集中式动环监控系统。"
    "运维团队按季度演练断电切换流程，演练记录归档保存三年。"
) * 3
# 实测：左填充偏差 0.000000000，右填充偏差 0.974795403。
# 1e-6 远低于故障量级，又给不同平台/精度留了余量。
_SELFTEST_MAX_BATCH_DRIFT = 1e-6

# 单次 forward 的最大条数。上游候选数不受本服务控制，激活内存随批大小线性增长，
# 这里是内存上限的硬约束。取 8 也保证了 2 条的批不变性自检落在同一次 forward 内。
_RERANK_MICRO_BATCH = 8


def _format_rerank_input(query: str, document: str) -> str:
    return (
        f"<Instruct>: {_RERANK_TASK}\n<Query>: {query}\n<Document>: {document}"
    )


class ReorderService:
    """文档重排序服务"""

    def __init__(self):
        self.LOCAL_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", r"D:\Hugging_Face\models\Qwen3-Reranker-0.6B")
        self.MODELSCOPE_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
        # Demo 固定使用 CPU，不随开发机的 CUDA 可用性改变运行设备。
        self.device = "cpu"
        self._model = None
        self._tokenizer = None
        self._token_true = None
        self._token_false = None
        self._prefix_tokens = None
        self._suffix_tokens = None

    async def _get_model(self):
        """懒加载模型实例。

        加载完成后立刻自检打分方向，不通过就抛异常而不是返回一个会打噪声分的模型。
        `RAG-013` 的根因是**失败不可见**：静默随机初始化 + `reorder_documents`
        照常返回 `success=True`，降级路径永不触发，日志和评测产物都看不出异常。
        这道自检把「静默错」变成「显式炸」，首次加载多一次 forward（CPU 约 0.6s）。
        """
        if self._model is None:
            actual_model_path = find_model_path(self.LOCAL_MODEL_PATH)
            logger.info(f"✅ 加载重排序模型：{actual_model_path}")

            # padding_side='left' 是硬要求：分数取 logits[:, -1, :]，即序列最后一个
            # 位置。右填充会让最后一个位置变成 pad token，取到的分布与文档内容无关，
            # 形态同样是「不报错，分数是噪声」。
            self._tokenizer = AutoTokenizer.from_pretrained(
                actual_model_path,
                padding_side="left",
                local_files_only=True,
            )
            # `output_loading_info=True` 让 transformers 把「哪些权重不是从
            # checkpoint 载入的」作为数据交回来，而不是只印一行日志。
            # `RAG-013` 的原始形态正是这行日志被淹没在启动输出里：
            # `score.weight | MISSING | newly initialized`，程序照常继续。
            model, loading_info = AutoModelForCausalLM.from_pretrained(
                actual_model_path,
                dtype=torch.float32,
                local_files_only=True,
                output_loading_info=True,
            )
            self._assert_all_weights_loaded(loading_info)
            model.eval()
            model.to(self.device)

            self._token_true = self._tokenizer.convert_tokens_to_ids("yes")
            self._token_false = self._tokenizer.convert_tokens_to_ids("no")
            self._prefix_tokens = self._tokenizer.encode(
                _RERANK_PREFIX, add_special_tokens=False
            )
            self._suffix_tokens = self._tokenizer.encode(
                _RERANK_SUFFIX, add_special_tokens=False
            )
            self._model = model

            try:
                self._assert_scoring_is_sane()
            except Exception:
                # 自检失败就不要留下一个半可用的实例：下次调用应重新加载并再次自检，
                # 而不是复用这个已知有问题的模型。
                self._model = None
                self._tokenizer = None
                raise

            logger.info(f"✅ 模型加载成功，使用设备：{self.device}")
        return self._model

    @staticmethod
    def _assert_all_weights_loaded(loading_info: Dict[str, Any]) -> None:
        """任何权重未从 checkpoint 载入都必须让加载失败。

        `RAG-013` 验收标准第 4 条：`score.weight MISSING` 一类加载告警必须让启动
        失败而不是继续。这里不解析日志文本，直接读 transformers 交回的结构化字段。
        """
        missing = sorted(loading_info.get("missing_keys") or [])
        mismatched = sorted(loading_info.get("mismatched_keys") or [])
        if missing or mismatched:
            raise RuntimeError(
                f"重排序模型权重未完整载入，拒绝启动。"
                f"随机初始化的权重会让打分变成噪声且**不抛异常**——"
                f"这正是 RAG-013 长期不可见的原因。"
                f"missing_keys={missing} mismatched_keys={mismatched}"
            )

    def _assert_scoring_is_sane(self) -> None:
        """加载自检：方向 + 批不变性。

        两条判据各管一类失败，不能互相替代（实测）：
          · 方向  -> 抓打分头未从 checkpoint 载入（`RAG-013` 的原始形态，
                     该情况下方向真的是随机的）；
          · 批不变性 -> 抓右填充。单对方向断言抓不住右填充：实测右填充下
                     间隔仍有 0.9700，方向"看起来"是对的。

        `chat template` 写错则两条都抓不住（实测丢掉官方 suffix 后间隔仍有
        0.7988），只能靠源码层把模板逐字钉死，见
        `tests/m3/test_reranker_scoring.py` 里的模板哈希断言。
        """
        self._assert_direction()
        self._assert_batch_invariance()

    def _assert_direction(self) -> None:
        """已知相关必须显著高于已知不相关。"""
        relevant, irrelevant = self._score_pairs(
            [
                _format_rerank_input(_SELFTEST_QUERY, _SELFTEST_RELEVANT),
                _format_rerank_input(_SELFTEST_QUERY, _SELFTEST_IRRELEVANT),
            ]
        )
        margin = relevant - irrelevant
        if margin <= _SELFTEST_MIN_MARGIN:
            raise RuntimeError(
                f"重排序模型打分自检未通过：已知相关 {relevant:.6f} 未显著高于"
                f"已知不相关 {irrelevant:.6f}（间隔 {margin:+.6f} <= "
                f"{_SELFTEST_MIN_MARGIN}）。打分没有相关性信号，拒绝使用该模型。"
                f"请检查 checkpoint 是否完整、是否被按序列分类模型加载"
                f"（该系列没有可加载的标量打分头）。详见 RAG-013。"
            )
        logger.info(
            f"✅ 打分方向自检通过：相关 {relevant:.6f} / 不相关 {irrelevant:.6f}"
            f"（间隔 {margin:+.6f}）"
        )

    def _assert_batch_invariance(self) -> None:
        """同一文档的分数不得取决于同批还有谁。

        分数取自 `logits[:, -1, :]`，即序列最后一个位置。左填充下短序列的末位
        仍是真实 token；右填充下会变成 pad token，取到的分布与文档内容无关，
        且只在「同批有更长序列」时才发生——单条打分时看不出来。

        长短两条必须落在同一次 forward，所以直接调 `_score_batch` 而不走
        `_score_pairs` 的微批切分：切分把它们分到两批就没有「同批」可言，自检会
        变成恒真。
        """
        rendered = _format_rerank_input(_SELFTEST_QUERY, _SELFTEST_RELEVANT)
        alone = self._score_batch([rendered])[0]
        batched = self._score_batch(
            [
                rendered,
                _format_rerank_input(_SELFTEST_QUERY, _SELFTEST_LONG_FILLER),
            ]
        )[0]
        drift = abs(alone - batched)
        if drift > _SELFTEST_MAX_BATCH_DRIFT:
            raise RuntimeError(
                f"重排序模型批不变性自检未通过：同一文档单独打分 {alone:.9f}，"
                f"与一条更长文档同批时 {batched:.9f}，偏差 {drift:.9f} > "
                f"{_SELFTEST_MAX_BATCH_DRIFT:g}。分数受同批其他文档影响，"
                f"通常是填充方向错了（必须 padding_side='left'，因为分数取"
                f"序列最后一个位置）。拒绝使用该模型。详见 RAG-013。"
            )
        logger.info(f"✅ 批不变性自检通过：偏差 {drift:.9f}")

    def _score_pairs(self, rendered_pairs: List[str]) -> List[float]:
        """按官方口径打分，返回与输入同序的相关性分数（0~1）。

        按 `_RERANK_MICRO_BATCH` 分微批。候选条数由上游决定（实测 M3 数据集单条
        query 可达 30 条），一次性 forward 的激活内存随之线性增长，本机 5 GiB
        可用内存撑不住 —— 分批是内存上限的保证，不是优化。

        分微批不改变分数：左填充下末位永远是真 token，批组成对结果无影响，这正是
        `_assert_batch_invariance` 守的性质（实测左填充偏差 0.000000000）。
        """
        scores: List[float] = []
        for start in range(0, len(rendered_pairs), _RERANK_MICRO_BATCH):
            scores.extend(
                self._score_batch(rendered_pairs[start : start + _RERANK_MICRO_BATCH])
            )
        return scores

    @torch.no_grad()
    def _score_batch(self, rendered_pairs: List[str]) -> List[float]:
        """单次 forward 打一批分。批不变性自检直接调它，绕开微批切分。

        自检要验的是「同一条文档在不同批组成下分数是否漂移」，必须保证那两条落在
        同一次 forward 里。若走 `_score_pairs`，未来把 `_RERANK_MICRO_BATCH` 调成
        1 就会把自检变成恒真 —— 右填充的唯一探测器会被静默删掉。
        """
        tokenizer = self._tokenizer
        budget = (
            _RERANK_MAX_LENGTH - len(self._prefix_tokens) - len(self._suffix_tokens)
        )
        inputs = tokenizer(
            rendered_pairs,
            return_tensors=None,
            add_special_tokens=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=budget,
        )
        for index, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][index] = (
                self._prefix_tokens + ids + self._suffix_tokens
            )
        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # `logits_to_keep=1` 只对末位做 lm_head 投影。不传的话模型会物化
        # [batch, seq, 151669] 的 float32 张量：batch=30、seq=600 就是 10.17 GiB，
        # 本机直接 OOM（exit 137）。而我们只用末位那一行，其余全是白算白占。
        logits = self._model(**inputs, logits_to_keep=1).logits[:, -1, :]
        # 只在 yes / no 这两个 token 上归一化：分数才落在 0~1 且彼此可比。
        stacked = torch.stack(
            [logits[:, self._token_false], logits[:, self._token_true]], dim=1
        )
        return (
            torch.nn.functional.log_softmax(stacked, dim=1)[:, 1].exp().tolist()
        )


    @property
    async def model(self):
        """获取模型实例（懒加载）"""
        return await self._get_model()
    
    async def reorder_documents(
        self,
        query: str,
        documents: List[str | RetrievalCandidate],
        thinking_callback=None,
    ) -> Dict[str, Any]:
        """
        对文档进行重排序
        :param query: 查询语句
        :param documents: 文档列表
        :param thinking_callback: 思考过程回调函数
        :return: 包含重排序结果的字典，格式为：
                 {"success": bool, "documents": List[Dict], "error": str}
        """
        try:
            if not documents:
                return {
                    "success": True,
                    "documents": [],
                    "error": ""
                }
            
            if thinking_callback:
                await thinking_callback({
                    "type": "thinking",
                    "stage": "reorder",
                    "content": f"正在计算 {len(documents)} 个文档的相关性分数..."
                })
            
            # 模型只读取渲染文本，候选身份始终保留在适配器内。
            rendered_documents = [
                document.reranker_text
                if isinstance(document, RetrievalCandidate)
                else document
                for document in documents
            ]

            # 触发懒加载与打分自检；自检不通过会抛出，走下面的降级路径。
            await self.model
            # 一次 forward 打完整批。原先写 batch_size=1 是为绕开
            # `Cannot handle batch sizes > 1 if no padding token is defined.`，
            # 那个报错本身是 RAG-013 的症状——pad token 从 tokenizer 取即可。
            scores = self._score_pairs(
                [
                    _format_rerank_input(query, document)
                    for document in rendered_documents
                ]
            )
            scores = validate_rerank_scores(scores, len(documents))
            
            # 构建结果列表
            scored_documents = []
            for source, document, score in zip(
                documents, rendered_documents, scores
            ):
                scored_documents.append({
                    "source": source,
                    "document": document,
                    "similarity": float(score),
                })
                logger.info(f"【重排序服务】文档相似度分数: {score:.4f}")
            
            if thinking_callback:
                score_details = []
                for i, (doc, score) in enumerate(
                    zip(rendered_documents, scores), 1
                ):
                    score_details.append({
                        "index": i,
                        "score": round(float(score), 4),
                        "preview": doc[:100] + "..." if len(doc) > 100 else doc
                    })
                await thinking_callback({
                    "type": "thinking",
                    "stage": "reorder",
                    "content": f"已计算完成 {len(documents)} 个文档的相关性分数，按分数降序排序",
                    "details": {
                        "scores": score_details
                    }
                })
            
            # 按相似度分数降序排序
            sorted_docs = sorted(
                scored_documents,
                key=lambda item: item["similarity"],
                reverse=True,
            )
            result_documents = []
            for rank, item in enumerate(sorted_docs, 1):
                result_item = {
                    "document": item["document"],
                    "similarity": item["similarity"],
                }
                if isinstance(item["source"], RetrievalCandidate):
                    result_item["candidate"] = item["source"].observe(
                        StageObservation(
                            stage="rerank",
                            route="cross_encoder",
                            rank=rank,
                            raw_score=item["similarity"],
                            score_direction="higher_is_better",
                        )
                    )
                result_documents.append(result_item)
            logger.info(f"【重排序服务】文档重排序成功，返回 {len(sorted_docs)} 个文档")
            
            return {
                "success": True,
                "documents": result_documents,
                "error": ""
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"【重排序服务】重排序失败: {error_msg}")
            return {
                "success": False,
                "documents": [],
                "error": error_msg
            }

    @staticmethod
    async def format_reorder_result(sorted_docs: List[Dict]) -> str:
        """
        格式化重排序结果
        :param sorted_docs: 重排序后的文档列表
        :return: 格式化后的字符串
        """
        formatted_result = "重排序后的文档列表：\n"
        for i, doc in enumerate(sorted_docs, 1):
            formatted_result += f"{i}. 相似度: {doc.get('similarity', 0):.4f}\n"
            formatted_result += f"   内容: {doc.get('document', '')}\n\n"
        return formatted_result


# 全局重排序服务实例
reorder_service = ReorderService()
