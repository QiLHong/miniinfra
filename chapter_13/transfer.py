"""第 13 期核心：同步/异步 KV Block 传输与显式完成语义。"""

import time
from dataclasses import dataclass, field

import torch


TRANSFER_MODES = ("sync", "async")


@dataclass
class TransferTask:
    """保存一次整请求传输的方向、块映射、事件和计时。

    completed 表示复制已完成；源块释放与映射提交仍由管理器的 finish 执行。"""
    direction: str
    request_id: str
    mapping: dict
    submit_clock_ms: float
    start_event: object | None = None
    end_event: object | None = None
    submit_wall_ms: float = 0.0
    exposed_wait_ms: float = 0.0
    completed: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def blocks(self):
        """返回本次映射涉及的整块数量，包含尾块中的空余容量。"""
        return self.mapping["blocks"]


class KVTransferManager:
    """管理一个请求的 KV Block 在 GPU 缓存与 CPU 池之间的传输。

    两种模式使用相同的资源预留和完成处理，但 CUDA 提交方式不同：
    `sync` 在当前计算流上以 non_blocking=False 复制，并等待该流完成；
    `async` 先用事件确认源数据就绪，再在独立复制流上以 non_blocking=True
    提交复制，由调度器稍后通过 query/wait 检查或等待完成。流是 GPU 操作的
    有序队列，事件用于建立不同流之间的先后关系；独立流不保证实际发生重叠。

    非 CUDA 设备始终直接同步复制。无论哪种模式，提交传输都不等于转移
    Block 所有权；只有 finish 确认复制完成后，才提交相应的资源状态变更。
    """

    def __init__(self, device, mode):
        """检查传输模式；CUDA 设备上建立复制流和计时基准事件，CPU 路径无需 CUDA 事件。"""
        if mode not in TRANSFER_MODES:
            raise ValueError("transfer mode 必须是 %s" % (TRANSFER_MODES,))
        self.device = device
        self.mode = mode
        self.is_cuda = device.type == "cuda"
        self.copy_stream = None
        self.epoch_event = None
        self.tasks = []
        if self.is_cuda:
            current = torch.cuda.current_stream(device)
            self.copy_stream = torch.cuda.Stream(device=device)
            self.epoch_event = torch.cuda.Event(enable_timing=True)
            self.epoch_event.record(current)
            current.synchronize()

    def _submit(self, direction, gpu_cache, cpu_pool, request_id, clock_ms):
        """先预留目标资源，再按设备和模式提交复制，最后登记传输任务。

        异步分支用 source_ready 让复制流等待当前计算流已有操作，确保复制时源数据
        已经准备好。提交后保留两端资源，只有确认完成才能释放源块。"""
        if direction == "d2h":
            mapping = cpu_pool.prepare_swap_out(gpu_cache, request_id)
        elif direction == "h2d":
            mapping = cpu_pool.prepare_swap_in(gpu_cache, request_id)
        else:
            raise ValueError("未知传输方向: " + direction)

        task = TransferTask(direction, request_id, mapping, clock_ms)
        host_start = time.perf_counter()
        if not self.is_cuda:
            self._copy_blocks(task, gpu_cache, cpu_pool, non_blocking=False)
            task.completed = True
        elif self.mode == "sync":
            stream = torch.cuda.current_stream(self.device)
            task.start_event = torch.cuda.Event(enable_timing=True)
            task.end_event = torch.cuda.Event(enable_timing=True)
            task.start_event.record(stream)
            self._copy_blocks(task, gpu_cache, cpu_pool, non_blocking=False)
            task.end_event.record(stream)
            stream.synchronize()
            task.completed = True
        else:
            compute_stream = torch.cuda.current_stream(self.device)
            source_ready = torch.cuda.Event()
            # 事件排在当前计算流已有操作之后，标记复制源的计算何时结束。
            source_ready.record(compute_stream)
            # 只让复制流等待源就绪，不在此处阻塞主机；随后提交的复制才能安全读源数据。
            self.copy_stream.wait_event(source_ready)
            task.start_event = torch.cuda.Event(enable_timing=True)
            task.end_event = torch.cuda.Event(enable_timing=True)
            task.start_event.record(self.copy_stream)
            with torch.cuda.stream(self.copy_stream):
                self._copy_blocks(task, gpu_cache, cpu_pool, non_blocking=True)
            task.end_event.record(self.copy_stream)
        task.submit_wall_ms = (time.perf_counter() - host_start) * 1000
        self.tasks.append(task)
        return task

    @staticmethod
    def _copy_blocks(task, gpu_cache, cpu_pool, non_blocking):
        """沿 gpu_ids/cpu_ids 一一对应的映射逐块复制，根据 direction 决定源与目标。"""
        for gpu_id, cpu_id in zip(
            task.mapping["gpu_ids"], task.mapping["cpu_ids"]
        ):
            if task.direction == "d2h":
                cpu_pool.blocks[cpu_id].copy_(
                    gpu_cache.blocks[gpu_id], non_blocking=non_blocking
                )
            else:
                gpu_cache.blocks[gpu_id].copy_(
                    cpu_pool.blocks[cpu_id], non_blocking=non_blocking
                )

    def submit_swap_out(self, gpu_cache, cpu_pool, request_id, clock_ms):
        """提交 GPU→CPU 换出并返回任务；调用方之后还要确认完成并调用 finish。"""
        return self._submit(
            "d2h", gpu_cache, cpu_pool, request_id, clock_ms
        )

    def submit_swap_in(self, gpu_cache, cpu_pool, request_id, clock_ms):
        """提交 CPU→GPU 换入并返回任务；复制完成前请求不能恢复模型执行。"""
        return self._submit(
            "h2d", gpu_cache, cpu_pool, request_id, clock_ms
        )

    def query(self, task):
        """非阻塞查询任务完成状态；CUDA 分支检查末尾事件，不在这里释放资源。"""
        if task.completed:
            return True
        if not self.is_cuda:
            task.completed = True
        elif task.end_event.query():
            task.completed = True
        return task.completed

    def wait(self, task):
        """必要时等待任务末尾事件，并累计主机实际等待的毫秒数；已完成任务返回 0。"""
        if self.query(task):
            return 0.0
        start = time.perf_counter()
        task.end_event.synchronize()
        waited = (time.perf_counter() - start) * 1000
        task.exposed_wait_ms += waited
        task.completed = True
        return waited

    def finish(self, task, gpu_cache, cpu_pool, complete_clock_ms):
        """确认复制完成后转移块所有权，并生成传输统计记录。

        换出完成才能归还 GPU 源块，换入完成才能归还 CPU 源块；提前释放会导致
        尚在进行的复制读到被其他请求覆盖的数据。"""
        # 完成检查是资源转移的前提；仅仅提交了异步复制，还不能重用源块。
        if not self.query(task):
            raise RuntimeError("传输尚未完成，不能转移 Block 所有权")
        if task.direction == "d2h":
            cpu_pool.complete_swap_out(gpu_cache, task.request_id)
            event_type = "swap_out"
        else:
            cpu_pool.complete_swap_in(task.request_id)
            event_type = "swap_in"
        mapping = task.mapping
        device_ms = 0.0
        gpu_start_ms = None
        gpu_end_ms = None
        if self.is_cuda:
            device_ms = task.start_event.elapsed_time(task.end_event)
            gpu_start_ms = self.epoch_event.elapsed_time(task.start_event)
            gpu_end_ms = self.epoch_event.elapsed_time(task.end_event)
        exposed_wait_ms = (
            device_ms if self.mode == "sync" else task.exposed_wait_ms
        )
        result = {
            key: value for key, value in mapping.items()
            if key not in ("gpu_ids", "cpu_ids")
        }
        result.update({
            "type": event_type,
            "direction": task.direction,
            "transfer_mode": self.mode,
            "submit_clock_ms": task.submit_clock_ms,
            "complete_clock_ms": complete_clock_ms,
            "submit_wall_ms": task.submit_wall_ms,
            "device_ms": device_ms,
            "exposed_wait_ms": exposed_wait_ms,
            "gpu_start_ms": gpu_start_ms,
            "gpu_end_ms": gpu_end_ms,
            "wall_ms": (
                task.submit_wall_ms if self.mode == "sync"
                else task.exposed_wait_ms
            ),
            **task.metadata,
        })
        result["gb_per_second"] = (
            result["bytes"] / (device_ms / 1000) / 1e9 if device_ms else 0.0
        )
        return result
