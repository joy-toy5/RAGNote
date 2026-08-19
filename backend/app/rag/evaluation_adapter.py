"""将严格的生产检索 trace 接入 M3 QueryExecutor 契约。"""

from __future__ import annotations

from typing import Protocol

from app.evaluation.contracts import ExecutorDescriptor, QueryExecution
from app.evaluation.runner import EvalRequest, execution_from_trace
from app.rag.retrieval_contract import RetrievalTrace


class RagRetrievalService(Protocol):
    @property
    def descriptor(self) -> ExecutorDescriptor: ...

    async def get_retrieval_trace(
        self,
        query: str,
        *,
        query_id: str,
    ) -> RetrievalTrace: ...


class RagServiceFactory(Protocol):
    @property
    def descriptor(self) -> ExecutorDescriptor: ...

    def __call__(self, request: EvalRequest) -> RagRetrievalService: ...


class RagServiceQueryExecutor:
    """逐 Query 创建生产检索服务并执行严格 trace 校验。"""

    __slots__ = ("_descriptor", "_issued_services", "_service_factory")

    def __init__(
        self,
        service_factory: RagServiceFactory,
    ) -> None:
        if not callable(service_factory):
            raise TypeError("service_factory 必须可调用")
        descriptor = service_factory.descriptor
        if not isinstance(descriptor, ExecutorDescriptor):
            raise TypeError("service_factory descriptor 必须是 ExecutorDescriptor")
        self._descriptor = descriptor
        self._service_factory = service_factory
        self._issued_services: list[RagRetrievalService] = []

    @property
    def descriptor(self) -> ExecutorDescriptor:
        return self._descriptor

    async def execute(self, request: EvalRequest) -> QueryExecution:
        self._validate_request(request)
        service = self._service_factory(request)
        if service.descriptor != self._descriptor:
            raise ValueError("检索服务 descriptor 与执行器声明不一致")
        if any(service is issued for issued in self._issued_services):
            raise RuntimeError("service_factory 必须为每条 Query 返回独立服务实例")
        self._issued_services.append(service)
        trace = await service.get_retrieval_trace(
            request.query,
            query_id=request.query_id,
        )
        if not isinstance(trace, RetrievalTrace):
            raise RuntimeError("生产检索服务必须返回 RetrievalTrace")
        return execution_from_trace(
            trace,
            expected_query_id=request.query_id,
            expected_user_id=request.user_id,
            expected_index_version=request.index_version,
        )

    def _validate_request(self, request: EvalRequest) -> None:
        if request.index_version != self._descriptor.index_version:
            raise ValueError("request index_version 与执行器声明不一致")
        if (
            request.retrieval_config_sha256
            != self._descriptor.retrieval_config_sha256
        ):
            raise ValueError("request retrieval config attestation 不一致")
