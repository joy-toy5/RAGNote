import asyncio
import os
import threading

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.utils.config import chroma_config
from app.utils.factory import embed_model
from app.utils.path_tool import get_abstract_path
from app.core.logger_handler import logger

from .retrievers.hybrid_retriever import HybridRetriever
from .md5_manager import MD5Store
from .document_handler import DocumentProcessor
from app.utils.image_extractor import delete_image_directory, delete_user_all_images


def _clear_chroma_cache():
    """
    清除 ChromaDB SharedSystemClient 内部单例缓存，避免 KeyError。
    ChromaDB 在 0.5.x+ 引入了 SharedSystemClient，它内部维护了一个全局 _instance 字典。
    当同一个进程反复创建/删除 Chroma 实例时，会抛出 KeyError（因为缓存中的 client 已被销毁）。
    在初始化前主动清除缓存，可以避免此问题。
    """
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass


class VectorStoreService:
    """
    向量数据库服务（单例，线程安全初始化）。

    使用双重检查锁定（Double-Checked Locking）实现线程安全的单例模式。
    之所以需要单例，是因为 ChromaDB 客户端维护了内部的连接池和缓存，
    多个实例会导致资源冲突和不可预期的 KeyError。
    """
    _instance = None
    _initialized = False
    _init_lock = threading.Lock()

    def __new__(cls):
        # 第一重检查（无锁，性能优先）
        if cls._instance is None:
            with cls._init_lock:
                # 第二重检查（加锁后，确保线程安全）
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if VectorStoreService._initialized:
            return

        with VectorStoreService._init_lock:
            if VectorStoreService._initialized:
                return

            persist_dir = get_abstract_path(chroma_config['persist_directory'])
            # 在创建 Chroma 实例前清除缓存，避免残留的单例 client 导致 KeyError
            _clear_chroma_cache()

            try:
                self._init_chroma(persist_dir)
            except Exception as e:
                logger.exception(
                    "Chroma 初始化失败，持久化目录保持不变；"
                    f"请检查权限、文件锁或数据完整性: {e}"
                )
                raise

            VectorStoreService._initialized = True

    def _init_chroma(self, persist_dir: str):
        self.vectors_store = Chroma(
            collection_name=chroma_config['collection_name'],
            embedding_function=embed_model,
            persist_directory=persist_dir,
        )
        self.md5_store = MD5Store()
        self.hybrid_retriever = HybridRetriever(self.vectors_store)
        self.document_processor = DocumentProcessor(self.vectors_store, self.md5_store)

    @classmethod
    def for_explicit_target(
        cls,
        *,
        persist_directory: str,
        collection_name: str,
        embedding_function,
        top_k: int,
        fusion_weights: tuple[float, float] | list[float] | None = None,
    ) -> "VectorStoreService":
        """
        构造一个绕过单例、指向显式目标的只读检索实例（离线评测用）。

        与单例路径的区别，以及为什么必须有这个区别：
        - 不写 cls._instance / cls._initialized，因此不污染生产单例；
        - 不调用 _clear_chroma_cache()，因为清空 SharedSystemClient 缓存会影响
          同进程内其他仍然存活的 client；
        - 不构造 MD5Store / DocumentProcessor，所以任何误用的写入/摄取路径会
          直接 AttributeError 失败，而不是静默写进评测索引；
        - k 与 fusion_weights 显式传入 HybridRetriever，否则 retrieval_config
          里声明的 top_k / 权重与实际行为不一致，attestation 就是假的。

        调用方必须在用完后调用 close()：每次构造 Chroma 都会让 chromadb
        SharedSystemClient 的 refcount +2，GC 不会回收，只有 close() 会减。
        """
        if not isinstance(persist_directory, str) or not persist_directory.strip():
            raise ValueError("显式检索目标必须提供持久化目录")
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("显式检索目标必须提供 collection 名称")
        if embedding_function is None:
            raise ValueError("显式检索目标必须提供 embedding function")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("显式检索目标的 top_k 必须是正整数")

        instance = object.__new__(cls)
        instance.vectors_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory,
        )
        instance.md5_store = None
        instance.document_processor = None
        instance.hybrid_retriever = HybridRetriever(
            instance.vectors_store,
            k=top_k,
            fusion_weights=fusion_weights,
        )
        return instance

    def close(self) -> None:
        """
        释放底层 chromadb client。

        只有显式构造的实例才应该被关闭；生产单例的 client 由进程生命周期管理，
        关闭它会让后续所有检索失效。
        """
        if VectorStoreService._instance is self:
            raise RuntimeError("不能关闭生产单例的向量库 client")
        store = getattr(self, "vectors_store", None)
        client = getattr(store, "_client", None)
        if client is None:
            return
        try:
            client.close()
        except Exception as e:
            logger.warning(f"【向量数据库】关闭显式 client 时出错: {e}")

    async def get_bm25_retriever(self, user_id: str):
        return await self.hybrid_retriever.get_bm25_retriever(user_id)

    async def _get_all_documents(self) -> list[Document]:
        return await self.hybrid_retriever._get_all_documents()

    async def get_retriever(self, query: str | None, user_id: str):
        return await self.hybrid_retriever.get_retriever(query, user_id)

    async def get_dynamic_weights(self, query: str = None):
        # 委托给实例：权重现在是 HybridRetriever 的实例状态（可被离线消融显式
        # 覆盖），不再是静态启发式。
        return await self.hybrid_retriever.get_dynamic_weights(query)

    async def check_md5_hex(self, md5_for_check: str, user_id: str = None) -> bool:
        return await self.md5_store.check_md5_hex(md5_for_check, user_id)

    async def get_md5_by_filename(self, user_id: str, filename: str) -> str | None:
        return await self.md5_store.get_md5_by_filename(user_id, filename)

    async def save_md5_hex(self, md5_hex: str, filename: str = None, original_filename: str = None, user_id: str = None):
        await self.md5_store.save_md5_hex(md5_hex, filename, original_filename, user_id)

    def save_md5_hex_sync(self, md5_hex: str, filename: str = None, original_filename: str = None, user_id: str = None):
        self.md5_store.save_md5_hex_sync(md5_hex, filename, original_filename, user_id)

    async def delete_user_documents(self, user_id: str):
        """
        删除指定用户的所有文档（包括MD5记录）
        :param user_id: 用户ID
        """
        try:
            await self.delete_user_md5(user_id, delete_documents=True)
        except Exception as e:
            logger.error(f"【向量数据库】删除用户 {user_id} 的文档时出错: {e}")
            raise

    async def delete_user_md5(self, user_id: str, delete_documents: bool = True):
        """
        删除指定用户的MD5记录
        :param user_id: 用户ID
        :param delete_documents: 是否同时删除向量数据库中的文档（默认True）
        """
        try:
            if delete_documents:
                await asyncio.to_thread(
                    self.vectors_store.delete,
                    where={"user_id": user_id}
                )
                await asyncio.to_thread(delete_user_all_images, user_id)
                logger.info(f"【向量数据库】已删除用户 {user_id} 的所有文档和图片")

            await self.md5_store.delete_user_md5(user_id)
        except Exception:
            logger.exception(f"【向量数据库】删除用户 {user_id} 的MD5记录时出错")
            raise

    async def delete_by_filename(self, user_id: str, filename: str, delete_documents: bool = True):
        """
        通过文件名删除MD5记录及其对应的知识库内容
        :param user_id: 用户ID
        :param filename: 要删除的文件名
        :param delete_documents: 是否同时删除向量数据库中的对应文档（默认True）
        :return: 是否成功删除
        """
        try:
            md5_to_delete = await self.md5_store.get_md5_by_filename(
                user_id, filename
            )
            if md5_to_delete is None:
                logger.warning(f"【向量数据库】文件 {filename} 不存在于用户 {user_id} 的MD5记录中")
                return False

            if delete_documents:
                where_clause = {"$and": [{"user_id": user_id}, {"md5": md5_to_delete}]}
                await asyncio.to_thread(
                    self.vectors_store.delete,
                    where=where_clause
                )
                await asyncio.to_thread(
                    delete_image_directory, user_id, md5_to_delete
                )
                logger.info(f"【向量数据库】已删除用户 {user_id} 中文件 {filename} 对应的文档和图片")

            deleted_md5 = await self.md5_store.delete_by_filename(user_id, filename)
            if deleted_md5 is None:
                raise RuntimeError("文件的MD5记录在删除前发生变化")
            logger.info(f"【向量数据库】已删除用户 {user_id} 的文件 {filename} 的MD5记录")
            return True

        except Exception:
            logger.exception(f"【向量数据库】删除用户 {user_id} 的文件 {filename} 时出错")
            raise

    async def delete_single_md5(self, user_id: str, md5_to_delete: str, delete_documents: bool = True):
        """
        删除单个MD5记录及其对应的知识库内容
        :param user_id: 用户ID
        :param md5_to_delete: 要删除的MD5值
        :param delete_documents: 是否同时删除向量数据库中的对应文档（默认True）
        :return: 是否成功删除
        """
        try:
            md5_info = await self.md5_store.get_md5_info(user_id, md5_to_delete)
            if md5_info is None:
                logger.warning(f"【向量数据库】MD5记录 {md5_to_delete} 不存在")
                return False

            if delete_documents:
                where_clause = {"$and": [{"user_id": user_id}, {"md5": md5_to_delete}]}
                await asyncio.to_thread(
                    self.vectors_store.delete,
                    where=where_clause
                )
                await asyncio.to_thread(
                    delete_image_directory, user_id, md5_to_delete
                )
                logger.info(f"【向量数据库】已删除用户 {user_id} 中MD5为 {md5_to_delete} 的文档和图片")

            deleted = await self.md5_store.delete_single_md5(user_id, md5_to_delete)
            if not deleted:
                raise RuntimeError("MD5记录在删除前发生变化")
            logger.info(f"【向量数据库】已删除用户 {user_id} 的MD5记录: {md5_to_delete}")
            return True

        except Exception:
            logger.exception(f"【向量数据库】删除用户 {user_id} 的MD5记录 {md5_to_delete} 时出错")
            raise

    async def get_md5_info(self, user_id: str, md5_value: str):
        """
        获取MD5对应的文档信息
        :param user_id: 用户ID
        :param md5_value: MD5值
        :return: MD5信息字典，不存在返回None
        """
        try:
            return await self.md5_store.get_md5_info(user_id, md5_value)
        except Exception as e:
            logger.error(f"【向量数据库】获取MD5信息 {md5_value} 时出错: {e}")
            return None

    async def get_all_md5_records(self, user_id: str):
        """
        获取用户的所有MD5记录
        :param user_id: 用户ID
        :return: MD5记录列表
        """
        try:
            records = await self.md5_store.get_all_md5_records(user_id)
            logger.info(f"【向量数据库】获取用户 {user_id} 的MD5记录，共 {len(records)} 条")
            return records
        except Exception as e:
            logger.error(f"【向量数据库】获取用户 {user_id} 的MD5记录时出错: {e}")
            return []

    async def get_user_documents(self, user_id: str):
        """
        获取用户的知识库文档列表
        :param user_id: 用户ID，如果为None则获取所有文档
        :return: 文档信息列表，包含文件名、文档数量、预览等信息
        """
        try:
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError("查询用户文档必须提供有效的用户 ID")
            where_clause = {"user_id": user_id}
            all_docs = await asyncio.to_thread(
                self.vectors_store.get,
                include=['documents', 'metadatas'],
                where=where_clause
            )

            docs_info = {}

            for i, doc_id in enumerate(all_docs['ids']):
                metadata = all_docs['metadatas'][i] if i < len(all_docs['metadatas']) else {}
                content = all_docs['documents'][i] if i < len(all_docs['documents']) else ""

                # 优先使用 metadata 中保存的 original_filename（用户上传时的原始文件名）
                # 因为 source 可能存的是临时文件的完整路径（如 C:\Users\...\tmp123.pdf），
                # 而 original_filename 才是用户看到的文件名
                source = metadata.get('source', metadata.get('filename', 'unknown'))
                if isinstance(source, str) and '\\' in source:
                    source = os.path.basename(source)
                filename = metadata.get('original_filename', source)

                original_filename = metadata.get('original_filename', filename)
                if filename not in docs_info:
                    docs_info[filename] = {
                        'id': doc_id,
                        'filename': filename,
                        'original_filename': original_filename,
                        'user_id': metadata.get('user_id'),
                        'chunk_count': 0,
                        'preview': "",
                        'created_at': metadata.get('created_at')
                    }

                docs_info[filename]['chunk_count'] += 1

                if not docs_info[filename]['preview'] and content:
                    preview_length = 100
                    docs_info[filename]['preview'] = content[:preview_length] + ("..." if len(content) > preview_length else "")

            result = list(docs_info.values())
            logger.info(f"【向量数据库】获取用户 {user_id} 的知识库文档，共 {len(result)} 个文件")
            return result

        except Exception as e:
            logger.error(f"【向量数据库】获取用户 {user_id} 的知识库文档时出错: {e}")
            raise

    async def get_document_detail(self, user_id: str, filename: str):
        """
        获取文档的详细内容
        :param user_id: 用户ID
        :param filename: 文件名
        :return: 文档详情信息，包含完整内容、图片列表和每段文本与图片的对应关系
        """
        try:
            where_clause = {"user_id": user_id}
            all_docs = await asyncio.to_thread(
                self.vectors_store.get,
                include=['documents', 'metadatas'],
                where=where_clause
            )

            doc_info = None
            full_content = []
            chunk_count = 0
            all_images = set()
            doc_md5 = None
            chunks = []

            for i, doc_id in enumerate(all_docs['ids']):
                metadata = all_docs['metadatas'][i] if i < len(all_docs['metadatas']) else {}
                content = all_docs['documents'][i] if i < len(all_docs['documents']) else ""

                source = metadata.get('source', metadata.get('filename', ''))
                if isinstance(source, str):
                    source_name = os.path.basename(source)
                else:
                    source_name = str(source)
                original_filename = metadata.get('original_filename', '')

                # 同时匹配 source 和 original_filename，兼容不同切片方式写入的 metadata
                if source_name == filename or original_filename == filename:
                    if not doc_info:
                        doc_info = {
                            'id': doc_id,
                            'filename': filename,
                            'user_id': metadata.get('user_id'),
                            'chunk_count': 0,
                            'content': "",
                            'images': [],
                            'md5': metadata.get('md5'),
                            'created_at': metadata.get('created_at')
                        }
                        doc_md5 = metadata.get('md5')
                    chunk_count += 1
                    full_content.append(content)

                    # 从 metadata 中取出该 chunk 关联的图片文件名列表，
                    # 拼接成可供前端直接请求的 URL 路径（由 knowledge_router 中的图片路由处理）
                    image_paths = metadata.get('image_paths', [])
                    chunk_images = []
                    if isinstance(image_paths, list):
                        for img_name in image_paths:
                            img_url = f"/knowledge/image/{doc_md5}/{img_name}"
                            all_images.add(img_url)
                            chunk_images.append(img_url)

                    chunks.append({
                        'chunk_id': doc_id,
                        'index': len(chunks),
                        'content': content,
                        'page': metadata.get('page'),
                        'images': chunk_images,
                    })

            if doc_info:
                doc_info['chunk_count'] = chunk_count
                doc_info['content'] = '\n'.join(full_content)
                doc_info['images'] = sorted(all_images)
                doc_info['chunks'] = chunks

            logger.info(f"【向量数据库】获取文档详情: {filename}，chunk数量: {chunk_count}，图片数量: {len(all_images)}")
            return doc_info

        except Exception as e:
            logger.error(f"【向量数据库】获取文档详情 {filename} 时出错: {e}")
            raise

    async def get_document_chunks(self, user_id: str, filename: str):
        """
        获取文档的所有切片信息
        :param user_id: 用户ID
        :param filename: 文件名
        :return: 切片列表信息，包含图片列表
        """
        try:
            where_clause = {"user_id": user_id}
            all_docs = await asyncio.to_thread(
                self.vectors_store.get,
                include=['documents', 'metadatas'],
                where=where_clause
            )

            chunks = []
            chunk_index = 0

            for i, doc_id in enumerate(all_docs['ids']):
                metadata = all_docs['metadatas'][i] if i < len(all_docs['metadatas']) else {}
                content = all_docs['documents'][i] if i < len(all_docs['documents']) else ""

                source = metadata.get('source', metadata.get('filename', ''))
                if isinstance(source, str):
                    source_name = os.path.basename(source)
                else:
                    source_name = str(source)
                original_filename = metadata.get('original_filename', '')

                if source_name == filename or original_filename == filename:
                    doc_md5 = metadata.get('md5', '')
                    # 解析图片路径：从 metadata 中拿到图片文件名列表，拼接为前端可用的API URL
                    image_paths = metadata.get('image_paths', [])
                    if isinstance(image_paths, list):
                        images = [f"/knowledge/image/{doc_md5}/{img}" for img in image_paths]
                    else:
                        images = []

                    chunks.append({
                        'chunk_id': doc_id,
                        'index': chunk_index,
                        'content': content,
                        'metadata': metadata,
                        'images': images,
                    })
                    chunk_index += 1

            result = {
                'filename': filename,
                'total_chunks': len(chunks),
                'chunks': chunks
            }

            logger.info(f"【向量数据库】获取文档切片: {filename}，共 {len(chunks)} 个切片")
            return result

        except Exception as e:
            logger.error(f"【向量数据库】获取文档切片 {filename} 时出错: {e}")
            raise

    # 以下方法将参数透传给 DocumentProcessor，使其能获取 md5 和 user_id 用于多模态PDF加载
    async def get_file_document(
        self,
        read_path: str,
        md5: str = None,
        user_id: str = None,
        source_filename: str = None,
    ) -> list[Document]:
        return await self.document_processor.get_file_document(
            read_path,
            md5,
            user_id,
            source_filename,
        )

    def get_file_document_sync(
        self,
        read_path: str,
        md5: str = None,
        user_id: str = None,
        source_filename: str = None,
    ) -> list[Document]:
        return self.document_processor.get_file_document_sync(
            read_path,
            md5,
            user_id,
            source_filename,
        )

    def split_documents_sync(self, documents: list[Document]) -> list[Document]:
        return self.document_processor.split_documents_sync(documents)

    async def get_document(self, files: list = None, user_id: str = None, progress_callback=None):
        await self.document_processor.get_document(files, user_id, progress_callback)


if __name__ == '__main__':
    async def main():
        store = VectorStoreService()
        await store.get_document()

        retriever = await store.get_retriever(None, "debug-user")
        results = await retriever.ainvoke('扫地')
        print(f"检索结果数量: {len(results)}")
        for result in results:
            print(result)

    asyncio.run(main())
