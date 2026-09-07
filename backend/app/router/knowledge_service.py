import asyncio
import base64
import time
import magic
import os
from typing import AsyncGenerator, List
from dataclasses import dataclass
from concurrent.futures import Future

from fastapi import HTTPException, UploadFile

from app.core.logger_handler import logger
from app.rag.vector_store import VectorStoreService
from app.rag.indexing_service import UploadIndexingService
from app.rag.task_queue import TaskQueue, QueueClosed
from app.rag.upload_runtime import UploadLease, upload_runtime
from app.rag.sse_models import SSEEvent, SliceResult



ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.md', '.pptx', '.docx'}
ALLOWED_MIME_TYPES = {
    'application/pdf', 'text/plain', 'text/markdown',
    'application/vnd.ms-powerpoint',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
}
MAX_FILE_SIZE = 20 * 1024 * 1024
MAX_FOLDER_SIZE = 200 * 1024 * 1024
MAX_UPLOAD_FILES = 100
UPLOAD_WAIT_TIMEOUT = 300.0


@dataclass
class ProcessingState:
    total_files: int = 0
    total_valid: int = 0
    sliced_count: int = 0
    written_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    slice_success_count: int = 0

    def current_progress(self) -> int:
        if self.total_valid == 0:
            return 0
        slice_progress = (self.sliced_count / self.total_valid) * 60
        write_progress = (self.written_count / self.total_valid) * 40
        return int(min(99, slice_progress + write_progress))


def _sync_slice_file(
    file_content: bytes,
    filename: str,
    file_index: int,
    user_id: str,
    media_type: str,
    queue: TaskQueue,
    store: VectorStoreService,
):
    """线程只负责 stage/parse；关闭后不再继续下一阶段或堵住 put。"""
    try:
        if queue.closed:
            return
        try:
            indexer = UploadIndexingService(store)
            staged = indexer.stage_upload(
                file_content, filename=filename, user_id=user_id, media_type=media_type,
            )
            if queue.closed:
                return
            prepared = indexer.prepare_upload_sync(staged)
            result = SliceResult.success_result(
                file_index=file_index, filename=filename,
                documents=list(prepared.documents), md5=prepared.legacy_md5,
                prepared_upload=prepared,
            )
        except Exception as exc:
            logger.error(f"【SSE上传】切片文件 {filename} 时出错: {exc}")
            result = SliceResult.error_result(file_index=file_index, filename=filename, error=str(exc))
        queue.put(result)
    except QueueClosed:
        # 已解析的 blob 保留；放弃通知不等于撤销已经发生的持久化。
        return


class KnowledgeService:
    """知识库管理服务"""

    async def handle_add_vector_single(self, file: UploadFile, user_id: str) -> str:
        """处理添加单个向量逻辑"""
        store = VectorStoreService()

        if file.size is not None and file.size > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="文件大小不能超过20MB")

        content = await file.read()
        mime = magic.Magic(mime=True)
        file_type = mime.from_buffer(content)

        file_extension = os.path.splitext(file.filename)[1].lower()

        if file_type not in ALLOWED_MIME_TYPES and file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"文件类型不支持，目前支持PDF、TXT、Markdown、PPTX、DOCX文件类型。检测到的文件类型: {file_type}，扩展名: {file_extension}"
            )

        await UploadIndexingService(store).index_upload(
            content,
            filename=file.filename,
            user_id=user_id,
            media_type=file_type,
        )
        return file.filename

    async def handle_add_vector_multiple(self, files: List[UploadFile], user_id: str) -> List[str]:
        """处理添加多个向量逻辑"""
        total_size = 0
        for file in files:
            total_size += file.size or 0

        if total_size > MAX_FOLDER_SIZE:
            raise HTTPException(status_code=400, detail="文件总大小不能超过200MB")

        start_time = time.time()
        results = []
        for file in files:
            try:
                await self.handle_add_vector_single(file, user_id)
                results.append(file.filename)
            except Exception as e:
                logger.error(f"【添加向量】处理文件 {file.filename} 时出错: {e}")
                raise

        end_time = time.time()
        logger.info(f"【添加向量】耗时: {end_time - start_time:.2f}秒，处理文件数: {len(results)}")

        return results

    def _yield_start_event(self, total_files: int) -> str:
        """SSE 事件：开始处理，通知前端文件总数"""
        return SSEEvent(
            event_type='start', total_files=total_files, message='开始处理文件...', progress=0
        ).to_sse()

    def _yield_size_error_event(self) -> str:
        """SSE 事件：文件总大小超限错误"""
        return SSEEvent(
            event_type='error', message='文件总大小不能超过200MB',
            error_message='文件总大小不能超过200MB'
        ).to_sse()

    def _yield_file_size_error_event(
        self, current_index: int, total_files: int, filename: str,
        file_size: int, failed_count: int
    ) -> str:
        """SSE 事件：单个文件大小超限错误"""
        return SSEEvent(
            event_type='error', file_index=current_index, total_files=total_files,
            filename=filename, step='validation',
            message=f'文件 {filename} 超过单文件大小限制',
            error_message=f'当前 {file_size / (1024 * 1024):.1f}MB，最大 20MB',
            progress=int(current_index / total_files * 100),
            failed_count=failed_count
        ).to_sse()

    def _yield_validation_error_event(
        self, current_index: int, total_files: int, filename: str,
        file_type: str, file_extension: str, failed_count: int
    ) -> str:
        """SSE 事件：单个文件 MIME 类型验证失败"""
        return SSEEvent(
            event_type='error', file_index=current_index, total_files=total_files,
            filename=filename, step='validation',
            message=f'文件 {filename} 类型不支持',
            error_message=f'文件类型: {file_type}，扩展名: {file_extension}',
            progress=int(current_index / total_files * 100),
            failed_count=failed_count
        ).to_sse()

    def _yield_slicing_completed_event(self, result: SliceResult, state: ProcessingState) -> str:
        """SSE 事件：单个文件多线程切片完成，准备写入向量库"""
        return SSEEvent(
            event_type='slicing_completed', file_index=result.file_index,
            total_files=state.total_files, filename=result.filename,
            chunk_count=result.chunk_count, step='slicing',
            message=f'文件 {result.filename} 切片完成，共 {result.chunk_count} 个切片',
            progress=state.current_progress(),
            success_count=state.success_count, failed_count=state.failed_count,
            slice_success_count=state.slice_success_count
        ).to_sse()

    def _yield_writing_event(self, result: SliceResult, state: ProcessingState) -> str:
        """SSE 事件：开始将切片结果写入向量数据库"""
        return SSEEvent(
            event_type='writing', file_index=result.file_index,
            total_files=state.total_files, filename=result.filename,
            step='writing', message=f'正在写入向量 {result.filename}...',
            progress=state.current_progress(),
            success_count=state.success_count, failed_count=state.failed_count,
            slice_success_count=state.slice_success_count
        ).to_sse()

    def _yield_completed_event(self, result: SliceResult, state: ProcessingState) -> str:
        """SSE 事件：单个文件全部处理完成（切片+写入成功）"""
        return SSEEvent(
            event_type='completed', file_index=result.file_index,
            total_files=state.total_files, filename=result.filename,
            step='completed', message=f'文件 {result.filename} 处理完成',
            progress=state.current_progress(),
            success_count=state.success_count, failed_count=state.failed_count,
            slice_success_count=state.slice_success_count
        ).to_sse()

    def _yield_write_error_event(self, result: SliceResult, state: ProcessingState, error: str) -> str:
        """SSE 事件：切片结果写入向量数据库时发生异常"""
        return SSEEvent(
            event_type='error', file_index=result.file_index,
            total_files=state.total_files, filename=result.filename,
            step='writing', message=f'文件 {result.filename} 写入失败',
            error_message=error,
            progress=state.current_progress(),
            success_count=state.success_count, failed_count=state.failed_count,
            slice_success_count=state.slice_success_count
        ).to_sse()

    def _yield_slice_error_event(self, result: SliceResult, state: ProcessingState) -> str:
        """SSE 事件：单个文件切片阶段失败（文件损坏/格式不支持等）"""
        return SSEEvent(
            event_type='error', file_index=result.file_index,
            total_files=state.total_files, filename=result.filename,
            step='slicing', message=f'文件 {result.filename} 切片失败',
            error_message=result.error,
            progress=state.current_progress(),
            success_count=state.success_count, failed_count=state.failed_count,
            slice_success_count=state.slice_success_count
        ).to_sse()

    def _yield_finish_event(self, start_time: float, total_files: int, success_count: int, failed_count: int) -> str:
        """SSE 事件：所有文件处理结束，汇总统计信息"""
        total_time = round(time.time() - start_time, 2)
        return SSEEvent(
            event_type='finish', total_files=total_files,
            success_count=success_count, failed_count=failed_count,
            message=f'处理完成，耗时 {total_time} 秒', progress=100
        ).to_sse()

    async def _validate_and_read_files(
        self, files: List[UploadFile]
    ) -> tuple[List[dict], List[str], int]:
        """
        阶段1: 读取文件内容并验证总大小
        阶段2: 逐一验证文件 MIME 类型
        返回 (有效文件列表, SSE错误事件列表, 总文件数)
        """
        total_files = len(files)
        if total_files > MAX_UPLOAD_FILES:
            raise HTTPException(status_code=400, detail=f"单批上传不能超过{MAX_UPLOAD_FILES}个文件")
        total_size = 0
        files_content = []
        error_events: List[str] = []
        failed_count = 0

        for current_index, file in enumerate(files, start=1):
            content = await file.read(min(MAX_FILE_SIZE + 1, MAX_FOLDER_SIZE - total_size + 1))
            file_size = len(content)
            total_size += file_size
            await file.seek(0)
            if total_size > MAX_FOLDER_SIZE:
                return [], [self._yield_size_error_event()], total_files

            if file_size > MAX_FILE_SIZE:
                failed_count += 1
                error_events.append(self._yield_file_size_error_event(
                    current_index, total_files, file.filename, file_size, failed_count
                ))
                logger.warning(
                    f"【SSE上传】文件大小验证失败: {file.filename}，"
                    f"大小: {file_size / (1024 * 1024):.2f}MB，限制: 20MB"
                )
                continue

            files_content.append({
                'file': file,
                'content': content,
                'file_index': current_index
            })

        mime = magic.Magic(mime=True)
        valid_files = []

        for file_info in files_content:
            file = file_info['file']
            content = file_info['content']
            current_index = file_info['file_index']
            file_type = mime.from_buffer(content)
            file_extension = os.path.splitext(file.filename)[1].lower()

            if file_type not in ALLOWED_MIME_TYPES and file_extension not in ALLOWED_EXTENSIONS:
                failed_count += 1
                error_events.append(self._yield_validation_error_event(
                    current_index, total_files, file.filename,
                    file_type, file_extension, failed_count
                ))
                logger.warning(f"【SSE上传】文件类型验证失败: {file.filename}，检测到类型: {file_type}，扩展名: {file_extension}")
            else:
                valid_files.append({
                    'content': content,
                    'filename': file.filename,
                    'file_index': current_index,
                    'media_type': file_type,
                })
                logger.debug(f"【SSE上传】文件类型验证通过: {file.filename}")

        return valid_files, error_events, total_files

    def _start_slicing(
        self, valid_files: List[dict], user_id: str, lease: UploadLease,
        store: VectorStoreService,
    ) -> list[Future]:
        """共享池最多四个解析线程；批次文件数与并行批次数均已限额。"""
        return [
            lease.submit(
                _sync_slice_file, info['content'], info['filename'], info['file_index'],
                user_id, info['media_type'], lease.queue, store,
            )
            for info in valid_files
        ]

    async def _next_slice_result(self, queue: TaskQueue, futures: list[Future]) -> SliceResult:
        deadline = time.monotonic() + UPLOAD_WAIT_TIMEOUT
        while True:
            try:
                return await queue.get_async(timeout=min(0.1, max(0, deadline - time.monotonic())))
            except TimeoutError:
                for future in futures:
                    if future.done():
                        # 暴露未投递结果的异常，不能被空队列轮询吞掉。
                        future.result()
                if all(future.done() for future in futures) and queue.empty():
                    raise RuntimeError("解析线程已结束，但缺少文件处理结果") from None
                if time.monotonic() >= deadline:
                    raise TimeoutError("等待解析超时；已开始的同步操作可能仍在执行，请勿自动重试") from None

    async def _process_slice_results(
        self, lease: UploadLease, futures: list[Future], store: VectorStoreService,
        state: ProcessingState,
    ) -> AsyncGenerator[str, None]:
        """异步消费解析结果；写入一旦开始，断开也保留其真实执行句柄。"""
        indexer = UploadIndexingService(store)
        while state.written_count < state.total_valid:
            result = await self._next_slice_result(lease.queue, futures)
            try:
                state.sliced_count += 1
                if not result.success:
                    state.written_count += 1
                    state.failed_count += 1
                    yield self._yield_slice_error_event(result, state)
                    continue

                state.slice_success_count += 1
                yield self._yield_slicing_completed_event(result, state)
                yield self._yield_writing_event(result, state)
                if lease.queue.closed:
                    raise QueueClosed("上传正在关闭，未开始本次索引写入")
                try:
                    if result.prepared_upload is None:
                        raise RuntimeError("切片结果缺少 durable upload 契约")
                    write = asyncio.create_task(indexer.persist_and_index(result.prepared_upload), name="upload-index")
                    lease.track_write(write)
                except Exception as exc:
                    state.written_count += 1
                    state.failed_count += 1
                    logger.error(f"【SSE上传】写入文件 {result.filename} 时出错: {exc}")
                    yield self._yield_write_error_event(result, state, str(exc))
                else:
                    # 只让出等待器，不取消含 to_thread 的索引协程。
                    _, pending = await asyncio.wait(
                        {write}, timeout=UPLOAD_WAIT_TIMEOUT
                    )
                    if pending:
                        raise TimeoutError("等待索引超时；写入仍可能执行，请勿自动重试")
                    try:
                        write.result()
                    except Exception as exc:
                        state.written_count += 1
                        state.failed_count += 1
                        logger.error(f"【SSE上传】写入文件 {result.filename} 时出错: {exc}")
                        yield self._yield_write_error_event(result, state, str(exc))
                    else:
                        state.written_count += 1
                        state.success_count += 1
                        yield self._yield_completed_event(result, state)
            finally:
                lease.queue.task_done()

    async def handle_add_vector_multiple_stream(
        self, files: List[UploadFile], user_id: str, *, prepared=None, lease=None,
    ) -> AsyncGenerator[str, None]:
        """保留低层上传流程；持久入口可传入独立输入与已占用的容量。"""
        total_files = len(files) if prepared is None else prepared[2]
        start_time = time.time()
        results = None
        yield self._yield_start_event(total_files)
        try:
            if total_files > MAX_UPLOAD_FILES:
                raise ValueError(f"单批上传不能超过{MAX_UPLOAD_FILES}个文件")
            # 先占容量，再将 UploadFile 读成 bytes，拒绝请求不排无限长队。
            if lease is None:
                lease = upload_runtime.acquire()
            valid_files, error_events, _ = (
                await self._validate_and_read_files(files) if prepared is None else prepared
            )
            for event in error_events:
                yield event
            if not valid_files:
                yield self._yield_finish_event(start_time, total_files, 0, total_files)
                return

            state = ProcessingState(
                total_files=total_files, total_valid=len(valid_files),
                failed_count=total_files - len(valid_files),
            )
            store = VectorStoreService()
            futures = self._start_slicing(valid_files, user_id, lease, store)
            results = self._process_slice_results(lease, futures, store, state)
            async for event in results:
                yield event
            yield self._yield_finish_event(start_time, total_files, state.success_count, state.failed_count)
        except Exception as exc:
            logger.error(f"【SSE上传】批次执行中断: {exc}")
            yield SSEEvent(event_type='error', message='上传处理中断', error_message=str(exc)).to_sse()
        finally:
            try:
                if results is not None:
                    await results.aclose()
            finally:
                if lease is not None:
                    lease.close()

    def _calculate_progress(self, sliced_count: int, written_count: int, total: int) -> int:
        if total == 0:
            return 0
        slice_progress = (sliced_count / total) * 60
        write_progress = (written_count / total) * 40
        return int(min(99, slice_progress + write_progress))

    async def clean_user_upload(self, user_id: str) -> None:
        """处理删除用户上传的所有向量逻辑"""
        store = VectorStoreService()
        await store.delete_user_documents(user_id)

    async def handle_clear_user_md5(self, user_id: str, delete_documents: bool = True) -> None:
        store = VectorStoreService()
        await store.delete_user_md5(user_id, delete_documents)
        if delete_documents:
            logger.info(f"【知识库】清空用户 {user_id} 的MD5记录和文档")
        else:
            logger.info(f"【知识库】清空用户 {user_id} 的MD5记录（保留知识库文档）")

    async def handle_delete_single_md5(self, user_id: str, md5_value: str, delete_documents: bool = True) -> bool:
        store = VectorStoreService()
        success = await store.delete_single_md5(user_id, md5_value, delete_documents)
        if success:
            logger.info(f"【知识库】删除用户 {user_id} 的MD5记录: {md5_value}")
        else:
            logger.warning(f"【知识库】删除用户 {user_id} 的MD5记录失败: {md5_value}")
        return success

    async def handle_delete_by_filename(self, user_id: str, filename: str, delete_documents: bool = True) -> bool:
        store = VectorStoreService()
        success = await store.delete_by_filename(user_id, filename, delete_documents)
        if success:
            logger.info(f"【知识库】删除用户 {user_id} 的文件: {filename}")
        else:
            logger.warning(f"【知识库】删除用户 {user_id} 的文件失败: {filename}")
        return success

    async def handle_get_md5_info(self, user_id: str, md5_value: str):
        store = VectorStoreService()
        return await store.get_md5_info(user_id, md5_value)

    async def handle_get_all_md5_records(self, user_id: str):
        store = VectorStoreService()
        return await store.get_all_md5_records(user_id)

    async def handle_get_user_knowledge(self, user_id: str) -> list:
        store = VectorStoreService()
        documents = await store.get_user_documents(user_id)
        logger.info(f"【知识库】获取用户 {user_id} 的知识库文档，共 {len(documents)} 个文件")
        return documents

    async def handle_get_document_detail(self, user_id: str, filename: str) -> dict:
        store = VectorStoreService()
        document = await store.get_document_detail(user_id, filename)
        if not document:
            raise HTTPException(status_code=404, detail=f"文档 {filename} 不存在")
        logger.info(f"【知识库】获取文档详情: {filename}")
        return document

    async def handle_get_document_chunks(self, user_id: str, filename: str) -> dict:
        store = VectorStoreService()
        chunks = await store.get_document_chunks(user_id, filename)
        if chunks['total_chunks'] == 0:
            raise HTTPException(status_code=404, detail=f"文档 {filename} 不存在或没有切片")
        logger.info(f"【知识库】获取文档切片: {filename}，共 {chunks['total_chunks']} 个切片")
        return chunks

    async def handle_get_batch_images(self, user_id: str, md5: str) -> dict:
        """
        一次性读取某个文档的所有提取图片，以 base64 data URL 的形式返回。
        这样前端可以一次请求拿到所有图片，然后根据 chunk 中的 image_paths 按需渲染，
        避免了每个图片单独发 HTTP 请求的性能开销（尤其适合移动端或图片较多的场景）。
        """
        from app.utils.image_extractor import get_image_path, get_image_storage_dir

        store = VectorStoreService()
        if not await store.get_md5_info(user_id, md5):
            raise HTTPException(status_code=404, detail="文档不存在")
        try:
            image_dir = get_image_storage_dir(user_id, md5, create=False)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="文档不存在") from exc
        if not os.path.isdir(image_dir):
            logger.warning(f"【知识库】图片目录不存在: {image_dir}")
            return {"md5": md5, "images": {}}

        images = {}
        try:
            for filename in sorted(os.listdir(image_dir)):
                try:
                    filepath = get_image_path(user_id, md5, filename)
                except ValueError:
                    continue
                if not filepath.is_file():
                    continue
                ext = filepath.suffix
                mime_map = {
                    '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                    '.tiff': 'image/tiff', '.tif': 'image/tiff',
                    '.bmp': 'image/bmp', '.gif': 'image/gif', '.webp': 'image/webp',
                }
                mime = mime_map.get(ext.lower(), 'application/octet-stream')
                with filepath.open("rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                images[filename] = f"data:{mime};base64,{b64}"
        except Exception as e:
            logger.error(f"【知识库】读取批量图片失败: {e}")
            raise HTTPException(status_code=500, detail=f"读取图片失败: {e}")

        logger.info(f"【知识库】读取批量图片: {md5}，共 {len(images)} 张")
        return {"md5": md5, "images": images}


def get_knowledge_service() -> KnowledgeService:
    """获取知识库服务实例（用于依赖注入）"""
    return KnowledgeService()
