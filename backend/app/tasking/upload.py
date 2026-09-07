"""SSE上传的薄接入层：复用任务表与P0执行器，不提供重启恢复。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid

from fastapi import HTTPException

from app.core.task_registry import background_tasks
from app.rag.upload_runtime import UploadBusy, upload_runtime
from app.tasking import repository
from app.tasking.contracts import TaskSubmission

logger = logging.getLogger(__name__)


def _session_factory():
    # 仅在真实接单时取得应用会话；导入和离线测试不创建生产引擎。
    from app.db.db_config import AsyncSessionLocal
    return AsyncSessionLocal


def _submission(prepared, user_id):
    valid_files, _, total = prepared
    identity = [(item['filename'], hashlib.sha256(item['content']).hexdigest()) for item in valid_files]
    token = uuid.uuid4().hex
    return TaskSubmission(
        user_id=user_id, kind="knowledge.index", idempotency_key=f"upload:{token}",
        input_fingerprint=hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest(),
        input_ref=f"memory://upload/{token}", input_metadata={"total_files": total}, max_attempts=1,
    )


def _event(payload, task_id):
    return 'event: progress\ndata: ' + json.dumps(dict(payload, task_id=task_id), ensure_ascii=False) + '\n\n'


async def _save_result(sessions, task_id, user_id, **values):
    async with sessions() as db:
        await repository.finish_local_task(db, task_id, user_id, **values)
        await db.commit()


async def _run_upload(service, prepared, lease, user_id, ready, queue, disconnected):
    task_id = None
    producer = None
    sessions = None

    def publish(payload):
        if not disconnected.is_set():
            queue.put_nowait(_event(payload, task_id))

    async def fail(code, summary):
        if task_id is not None:
            try:
                await _save_result(sessions, task_id, user_id, status="failed", error_code=code, error_summary=summary)
            except Exception:
                logger.error("上传任务终态写入失败；请按任务ID检查记录")
        if not ready.done():
            ready.set_exception(HTTPException(status_code=503, detail="任务暂时无法接单"))
        publish(dict(event_type="error", message=summary, error_message=summary))

    try:
        sessions = _session_factory()
        async with sessions() as db:
            task, _ = await repository.create_or_get_task(db, _submission(prepared, user_id))
            task_id = task.task_id
            await db.commit()
        ready.set_result(task_id)
        async with sessions() as db:
            started = await repository.start_local_task(db, task_id, user_id)
            await db.commit()
        if not started:
            publish(dict(event_type="error", message="任务已取消或不再待执行"))
            return
        finish = None
        producer = service.handle_add_vector_multiple_stream([], user_id, prepared=prepared, lease=lease)
        async for event in producer:
            payload = json.loads(event.split('data: ', 1)[1])
            if payload['event_type'] == 'finish':
                finish = payload
            else:
                publish(payload)
        if finish is None:
            await fail("UPLOAD_FAILED", "上传处理中断，请检查任务状态")
            return
        failed = finish.get('failed_count', 0)
        succeeded = finish.get('success_count', 0)
        await _save_result(
            sessions, task_id, user_id, status="failed" if failed else "succeeded",
            result_ref="/knowledge/list" if succeeded else None,
            error_code="UPLOAD_FAILED" if failed else None,
            error_summary=f"上传完成：成功{succeeded}，失败{failed}" if failed else None,
        )
        publish(finish)
    except asyncio.CancelledError:
        await fail("SHUTDOWN_TIMEOUT", "停机等待超时，上传未确认完成；不会自动重试")
        raise
    except Exception:
        await fail("UPLOAD_FAILED", "上传失败，请检查任务状态后重新提交")
    finally:
        try:
            if producer is not None:
                await producer.aclose()
        finally:
            lease.close()
            queue.put_nowait(None)


async def _events(queue, disconnected):
    try:
        while (event := await queue.get()) is not None:
            yield event
    finally:
        # 只取消订阅；工作协程和输入仍由登记器持有。
        disconnected.set()


async def submit_upload(service, files, user_id):
    """返回已提交的稳定任务ID及进度订阅，不把HTTP连接作为执行拥有者。"""
    if not background_tasks.accepting:
        raise HTTPException(status_code=503, detail="正在停止服务，不再接受新任务")
    try:
        lease = upload_runtime.acquire()
    except UploadBusy as exc:
        raise HTTPException(status_code=503, detail="上传容量已满或正在停止服务") from exc
    scheduled = False
    try:
        prepared = await service._validate_and_read_files(files)
        ready = asyncio.get_running_loop().create_future()
        # 响应在接单提交期间断开，也观察失败通知，避免未取回Future异常。
        ready.add_done_callback(lambda future: None if future.cancelled() else future.exception())
        queue, disconnected = asyncio.Queue(), asyncio.Event()
        try:
            background_tasks.create(
                _run_upload(service, prepared, lease, user_id, ready, queue, disconnected), name="knowledge-upload",
            )
        except RuntimeError as exc:
            # 读取文件期间可能开始停机；登记器关闭被拒绝的协程，本层释放容量。
            raise HTTPException(status_code=503, detail="任务暂时无法接单") from exc
        scheduled = True
        task_id = await asyncio.shield(ready)
        return task_id, _events(queue, disconnected)
    finally:
        if not scheduled:
            lease.close()
