"""第 15 期独立使用的 Paged KV Cache。

Prefill Worker 导出逻辑连续 KV，Decode Worker 在自己的池中重新申请 Physical
Block。跨进程协议永远不传递只在单个 Worker 内有效的 Block ID。
"""

import heapq
import math

import torch


DTYPE_NAMES = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}
NAME_DTYPES = {value: key for key, value in DTYPE_NAMES.items()}


class PagedKVCache:
    """用固定大小的物理块保存 K/V，用请求自己的 Block Table 恢复逻辑顺序。

    逻辑 token 位置 p 对应表项 p//block_size，块内偏移为 p%block_size。
    物理块编号可以不连续；缓存容量、已分配块数和有效 token 数是不同的量。"""
    def __init__(self, config, block_size, max_blocks, device, dtype):
        """预分配 K/V 池并初始化空闲块、请求块表和有效长度。

        布局为 [层, K/V, 物理块, KV头, 块内token, 每头维度]；空闲块中的内容
        尚未初始化，只有已写入且位于请求有效长度内的位置才属于可读缓存。"""
        if block_size < 1 or max_blocks < 1:
            raise ValueError("block_size 和 max_blocks 必须大于 0")
        if dtype not in DTYPE_NAMES:
            raise ValueError("不支持的 KV dtype: %s" % dtype)
        self.config = config
        self.block_size = int(block_size)
        self.max_blocks = int(max_blocks)
        self.device = torch.device(device)
        self.dtype = dtype
        self.blocks = torch.empty(
            (
                config.num_hidden_layers,
                2,
                max_blocks,
                config.num_key_value_heads,
                block_size,
                config.head_dim,
            ),
            dtype=dtype,
            device=self.device,
        )
        self.free_block_ids = list(range(max_blocks))
        self.block_tables = {}
        self.sequence_lengths = {}
        self.peak_used_blocks = 0

    @property
    def bytes_per_token(self):
        """计算一个 token 在全部层中保存 K 和 V 所需的字节数，不包括尾块碎片。"""
        return (
            self.config.num_hidden_layers
            * 2
            * self.config.num_key_value_heads
            * self.config.head_dim
            * torch.empty((), dtype=self.dtype).element_size()
        )

    @property
    def bytes_per_block(self):
        """用单块元素数量乘 dtype 字节数，得到块容量；尾块未填满时仍占完整块。"""
        return self.bytes_per_token * self.block_size

    @property
    def used_blocks(self):
        """返回块池中已经分配的物理块数。"""
        return self.max_blocks - len(self.free_block_ids)

    def create_request(self, request_id):
        """为新请求建立空块表和零有效长度；实际物理块在追加数据时分配。"""
        if request_id in self.block_tables:
            raise ValueError("请求已经存在: " + request_id)
        self.block_tables[request_id] = []
        self.sequence_lengths[request_id] = 0

    def _ensure_blocks(self, request_id, token_count):
        """按 ceil(token_count/block_size) 补齐请求块表，并更新占用峰值。

        这里只保证容量，不写入 K/V，也不自动推进请求的有效长度。"""
        required = math.ceil(token_count / self.block_size)
        table = self.block_tables[request_id]
        while len(table) < required:
            if not self.free_block_ids:
                raise RuntimeError("Paged KV Cache 物理 Block 已耗尽")
            table.append(heapq.heappop(self.free_block_ids))
        self.peak_used_blocks = max(self.peak_used_blocks, self.used_blocks)

    def append(self, request_id, current_cache):
        """追加只包含本轮新 Token 的逐层 K/V。"""
        if not current_cache:
            raise ValueError("current_cache 不能为空")
        count = current_cache[0][0].shape[2]
        if count < 1:
            raise ValueError("追加 Token 数必须大于 0")
        start = self.sequence_lengths[request_id]
        end = start + count
        self._ensure_blocks(request_id, end)
        table = self.block_tables[request_id]
        expected = (
            1,
            self.config.num_key_value_heads,
            count,
            self.config.head_dim,
        )
        for layer, (key, value) in enumerate(current_cache):
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise ValueError("追加的 KV Cache 形状错误")
            source = 0
            position = start
            while position < end:
                block_id = table[position // self.block_size]
                # 余数给出块内位置；整除负责选块，两者合起来才能定位一个 token。
                offset = position % self.block_size
                # 限制本次复制在当前块内；输入片段跨块时由 while 循环继续处理剩余部分。
                length = min(end - position, self.block_size - offset)
                self.blocks[layer, 0, block_id, :, offset:offset + length, :].copy_(
                    key[0, :, source:source + length, :]
                )
                self.blocks[layer, 1, block_id, :, offset:offset + length, :].copy_(
                    value[0, :, source:source + length, :]
                )
                source += length
                position += length
        # 所有层写完后才发布新的逻辑长度，未写入的池容量仍不可作为有效历史读取。
        self.sequence_lengths[request_id] = end

    def read_layer(self, layer, request_id, length=None):
        """通过 Block Table 按逻辑顺序收集一层的 K/V，并裁掉尾块未使用的位置。

        返回连续逻辑视图所需的稠密张量；高级索引和 reshape 可能产生复制，
        因此这条教学路径并不等于直接在分页存储上执行融合 Attention 内核。"""
        if length is None:
            length = self.sequence_lengths[request_id]
        if length == 0:
            return None
        block_count = math.ceil(length / self.block_size)
        ids = torch.as_tensor(
            self.block_tables[request_id][:block_count],
            dtype=torch.long,
            device=self.device,
        )
        key = self.blocks[layer, 0, ids].permute(1, 0, 2, 3).reshape(
            1,
            self.config.num_key_value_heads,
            block_count * self.block_size,
            self.config.head_dim,
        )[:, :, :length, :]
        value = self.blocks[layer, 1, ids].permute(1, 0, 2, 3).reshape(
            1,
            self.config.num_key_value_heads,
            block_count * self.block_size,
            self.config.head_dim,
        )[:, :, :length, :]
        return key, value

    def dense(self, request_id):
        """按逻辑顺序收集请求的全部层 K/V，空缓存返回 None，供常规模型前向使用。"""
        length = self.sequence_lengths[request_id]
        if length == 0:
            return None
        return [
            self.read_layer(layer, request_id, length)
            for layer in range(self.config.num_hidden_layers)
        ]

    def external_shape(self, token_count):
        """给出外部序列化布局 [层, K/V, KV头, token, 每头维度]，不包含进程内物理块 ID。"""
        return (
            self.config.num_hidden_layers,
            2,
            self.config.num_key_value_heads,
            token_count,
            self.config.head_dim,
        )

    def export_chunk(self, request_id, start, token_count):
        """导出指定 token 区间的全部层 K/V 为 CPU 连续字节流。

        读取按逻辑顺序进行，传输内容与本进程的块编号无关；会发生设备到 CPU 的复制。"""
        end = start + token_count
        if start < 0 or end > self.sequence_lengths[request_id]:
            raise ValueError("导出的 Token 区间超出请求 Cache")
        output = torch.empty(
            self.external_shape(token_count), dtype=self.dtype, device="cpu"
        )
        for layer in range(self.config.num_hidden_layers):
            key, value = self.read_layer(layer, request_id, end)
            output[layer, 0].copy_(key[0, :, start:end, :])
            output[layer, 1].copy_(value[0, :, start:end, :])
        return output.contiguous().view(torch.uint8).numpy().tobytes()

    def import_chunk(self, request_id, payload, token_count):
        """先按 dtype 和形状验证字节长度，再还原逐层 K/V 并追加到本地请求。

        接收方重新分配物理块，不能直接使用发送方的块编号；字节校验和由上层协议负责。"""
        shape = self.external_shape(token_count)
        expected_bytes = math.prod(shape) * torch.empty(
            (), dtype=self.dtype
        ).element_size()
        if len(payload) != expected_bytes:
            raise ValueError(
                "外部 KV Payload 字节数错误: expected=%d actual=%d"
                % (expected_bytes, len(payload))
            )
        # 先按原始字节接收，再依据约定 dtype/shape 解释；这里不是逐元素数值类型转换。
        byte_tensor = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        source = byte_tensor.view(self.dtype).reshape(shape)
        current_cache = [
            (
                source[layer, 0].unsqueeze(0).to(self.device),
                source[layer, 1].unsqueeze(0).to(self.device),
            )
            for layer in range(self.config.num_hidden_layers)
        ]
        self.append(request_id, current_cache)

    def release(self, request_id):
        """移除请求的逻辑块表与长度，按本章的所有权规则归还可释放的物理块。

        只释放分配关系，不搬动其他请求的数据；共享前缀版本还要保留被引用的块。"""
        table = self.block_tables.pop(request_id)
        self.sequence_lengths.pop(request_id)
        for block_id in table:
            heapq.heappush(self.free_block_ids, block_id)

    def snapshot(self):
        """返回块池容量、占用与分配统计的可记录快照，供测试和实验报告检查资源变化。"""
        return {
            "block_size": self.block_size,
            "max_blocks": self.max_blocks,
            "used_blocks": self.used_blocks,
            "peak_used_blocks": self.peak_used_blocks,
            "bytes_per_token": self.bytes_per_token,
            "bytes_per_block": self.bytes_per_block,
            "pool_bytes": self.max_blocks * self.bytes_per_block,
            "sequence_lengths": dict(self.sequence_lengths),
            "block_tables": {
                key: list(value) for key, value in self.block_tables.items()
            },
        }
