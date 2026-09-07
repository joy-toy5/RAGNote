"""显式启动组合：模型和两个集合就绪后，主入口才能开放接单。

本检查点只迁移初始化；存储仍归进程生命周期所有，不提供写入隔离或停止证明。
"""


def initialize_rag_resources() -> None:
    """保持导入无资源构造；两个集合复用同一客户端和嵌入模型。"""
    from app.utils.factory import get_chat_model, get_embed_model, get_vision_model
    from app.rag.vector_store import VectorStoreService
    from app.services.note_service import note_service

    get_chat_model()
    embedding = get_embed_model()
    get_vision_model()
    vector_store = VectorStoreService()
    note_service.initialize_storage(
        client=vector_store.vectors_store._client,
        embedding_function=embedding,
    )
