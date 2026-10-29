"""Paged KV Cache 的物理 Block Pool、Block Table 与参考 Attention 路径。

每个物理 Block 同时保存模型全部层的 K/V。请求只保存逻辑 Block 到物理
Block ID 的映射；请求结束后物理块回到空闲池，不移动其他请求的数据。
"""

import heapq
import math
from collections import defaultdict

import torch
import torch.nn.functional as F

from qwen3_model import apply_rotary_embedding


class PagedKVCache:
    """用固定大小的物理块保存 K/V，用请求自己的 Block Table 恢复逻辑顺序。

    逻辑 token 位置 p 对应表项 p//block_size，块内偏移为 p%block_size。
    物理块编号可以不连续；缓存容量、已分配块数和有效 token 数是不同的量。"""
    def __init__(self, config, block_size, max_blocks, device, dtype):
        """预分配 K/V 池并初始化空闲块、请求块表和有效长度。

        布局为 [层, K/V, 物理块, KV头, 块内token, 每头维度]；空闲块中的内容
        尚未初始化，只有已写入且位于请求有效长度内的位置才属于可读缓存。"""
        if block_size < 1:
            raise ValueError("block_size 必须大于 0")
        if max_blocks < 1:
            raise ValueError("max_blocks 必须大于 0")
        self.config = config
        self.block_size = block_size
        self.device = device
        self.dtype = dtype
        self.max_blocks = max_blocks
        self.blocks = torch.empty(
            (config.num_hidden_layers, 2, max_blocks,
             config.num_key_value_heads, block_size, config.head_dim),
            dtype=dtype,
            device=device,
        )
        self.free_block_ids = list(range(max_blocks))
        self.block_tables = {}
        self.sequence_lengths = {}
        self.peak_used_blocks = 0
        self.allocation_count = 0
        self.reuse_count = 0
        self.release_count = 0
        self.ever_used_block_ids = set()

    @property
    def block_shape(self):
        """返回一个物理块包含的全部层、K/V、KV头、token槽位及每头维度。"""
        return (
            self.config.num_hidden_layers,
            2,
            self.config.num_key_value_heads,
            self.block_size,
            self.config.head_dim,
        )

    @property
    def bytes_per_block(self):
        """用单块元素数量乘 dtype 字节数，得到块容量；尾块未填满时仍占完整块。"""
        elements = math.prod(self.block_shape)
        return elements * torch.empty((), dtype=self.dtype).element_size()

    @property
    def used_block_count(self):
        """返回当前占用的物理块数量，用于容量约束与资源统计。"""
        return sum(len(table) for table in self.block_tables.values())

    @property
    def live_token_count(self):
        """对各请求的有效缓存长度求和；共享前缀按请求分别计数。"""
        return sum(self.sequence_lengths.values())

    @property
    def allocated_token_slots(self):
        """将占用块数换算为物理 token 槽位数，包含每个请求尾块的空余位置。"""
        return self.used_block_count * self.block_size

    @property
    def live_cache_bytes(self):
        """按已占用的整块计算缓存字节数，而不是仅统计有效 token。"""
        return self.used_block_count * self.bytes_per_block

    @property
    def pool_bytes(self):
        """返回整个预分配块池的容量字节数，包括当前空闲的物理块。"""
        return self.max_blocks * self.bytes_per_block

    @property
    def utilization(self):
        """计算有效 token 总数与物理槽位数之比，空池返回 0。

        启用共享前缀后，同一物理块可能被多个请求计入有效长度；该值此时不再是
        严格限制在 0 到 1 之间的物理填充率，需结合引用计数理解。"""
        slots = self.allocated_token_slots
        return self.live_token_count / slots if slots else 0.0

    def _new_block(self):
        """从空闲块池领取一个物理块，并更新首次分配或复用统计；容量耗尽时显式失败。"""
        if not self.free_block_ids:
            raise RuntimeError("Paged KV Cache 物理 Block 已耗尽")
        block_id = heapq.heappop(self.free_block_ids)
        if block_id in self.ever_used_block_ids:
            self.reuse_count += 1
        else:
            self.ever_used_block_ids.add(block_id)
            self.allocation_count += 1
        return block_id

    def _ensure_blocks(self, request_id, token_count):
        """按 ceil(token_count/block_size) 补齐请求块表，并更新占用峰值。

        这里只保证容量，不写入 K/V，也不自动推进请求的有效长度。"""
        # 整数向上取整：例如块长 4、有效长度 5，需要 2 块，第二块只用了 1 个槽位。
        required = (token_count + self.block_size - 1) // self.block_size
        table = self.block_tables[request_id]
        while len(table) < required:
            table.append(self._new_block())
        self.peak_used_blocks = max(self.peak_used_blocks, self.used_block_count)

    def store_prefill(self, request_id, request_cache):
        """把单请求的逐层连续 K/V 拆成物理块保存，并建立块表与有效长度。

        每层输入为 [1, KV头数, 提示长度, 每头维度]；尾块只复制实际存在的 token。"""
        if request_id in self.block_tables:
            raise ValueError("请求已经存在于 Block Pool: " + request_id)
        if not request_cache:
            raise ValueError("request_cache 不能为空")
        length = request_cache[0][0].shape[2]
        if length < 1:
            raise ValueError("Prefill Cache 长度必须大于 0")
        self.block_tables[request_id] = []
        self.sequence_lengths[request_id] = length
        self._ensure_blocks(request_id, length)
        table = self.block_tables[request_id]
        for layer_index, (key, value) in enumerate(request_cache):
            if key.shape != value.shape or key.shape[0] != 1:
                raise ValueError("单请求 Prefill Cache 形状错误")
            for logical_block, block_id in enumerate(table):
                start = logical_block * self.block_size
                end = min(start + self.block_size, length)
                count = end - start
                self.blocks[layer_index, 0, block_id, :, :count, :].copy_(
                    key[0, :, start:end, :]
                )
                self.blocks[layer_index, 1, block_id, :, :count, :].copy_(
                    value[0, :, start:end, :]
                )

    def dense_request_cache(self, request_id):
        """按逻辑顺序重建单请求 Cache，供教学版增量 Prefill 使用。"""
        if request_id not in self.block_tables:
            return None
        length = self.sequence_lengths[request_id]
        return [
            self.read_layer(layer_index, [request_id], length)
            for layer_index in range(self.config.num_hidden_layers)
        ]

    def append_prefill(self, request_id, request_cache):
        """把只包含当前 Prompt Chunk 的 K/V 追加到 Block Table。"""
        if not request_cache:
            raise ValueError("request_cache 不能为空")
        chunk_length = request_cache[0][0].shape[2]
        if chunk_length < 1:
            raise ValueError("Prefill Chunk 长度必须大于 0")
        if request_id not in self.block_tables:
            self.block_tables[request_id] = []
            self.sequence_lengths[request_id] = 0
        start = self.sequence_lengths[request_id]
        end = start + chunk_length
        self._ensure_blocks(request_id, end)
        table = self.block_tables[request_id]
        for layer_index, (key, value) in enumerate(request_cache):
            expected = (
                1, self.config.num_key_value_heads,
                chunk_length, self.config.head_dim,
            )
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise ValueError("Prefill Chunk Cache 形状错误")
            source = 0
            position = start
            while position < end:
                block_id = table[position // self.block_size]
                # 余数给出块内位置；整除负责选块，两者合起来才能定位一个 token。
                offset = position % self.block_size
                # 每次最多填到当前物理块末尾；剩余片段继续写下一块，因而支持跨块追加。
                count = min(end - position, self.block_size - offset)
                self.blocks[
                    layer_index, 0, block_id, :, offset:offset + count, :
                ].copy_(key[0, :, source:source + count, :])
                self.blocks[
                    layer_index, 1, block_id, :, offset:offset + count, :
                ].copy_(value[0, :, source:source + count, :])
                position += count
                source += count
        # 所有层写完后才发布新的逻辑长度，未写入的池容量仍不可作为有效历史读取。
        self.sequence_lengths[request_id] = end

    def prepare_append(self, request_ids):
        """为每个请求的下一个 token 预留块，返回本次写入的逻辑位置。

        此时长度仍是旧值；写完所有层后再由 finish_append 增加一次长度。"""
        positions = {}
        for request_id in request_ids:
            if request_id not in self.block_tables:
                raise KeyError("请求没有 Block Table: " + request_id)
            position = self.sequence_lengths[request_id]
            self._ensure_blocks(request_id, position + 1)
            positions[request_id] = position
        return positions

    def write_token(self, layer_index, request_id, key, value, position):
        """将某层的新 K/V 写入逻辑 position 对应的物理块和块内偏移。

        输入 K/V 为 [KV头数, 1, 每头维度]。一次调用只写一层，不推进全请求长度。"""
        if tuple(key.shape) != (
            self.config.num_key_value_heads, 1, self.config.head_dim
        ) or key.shape != value.shape:
            raise ValueError("Decode K/V 必须是 [num_kv_heads, 1, head_dim]")
        # 先用逻辑位置找到块表条目，再得到物理块 ID；不能把逻辑块号直接当物理地址。
        block_id = self.block_tables[request_id][position // self.block_size]
        # 余数给出块内位置；整除负责选块，两者合起来才能定位一个 token。
        offset = position % self.block_size
        self.blocks[layer_index, 0, block_id, :, offset, :].copy_(key[:, 0, :])
        self.blocks[layer_index, 1, block_id, :, offset, :].copy_(value[:, 0, :])

    def finish_append(self, request_ids):
        """在所有层的新 K/V 都写入后，将每个请求的有效长度统一增加 1。"""
        for request_id in request_ids:
            self.sequence_lengths[request_id] += 1

    def read_layer(self, layer_index, request_ids, logical_length):
        """通过 Block Table 按逻辑顺序收集一层的 K/V，并裁掉尾块未使用的位置。

        返回连续逻辑视图所需的稠密张量；高级索引和 reshape 可能产生复制，
        因此这条教学路径并不等于直接在分页存储上执行融合 Attention 内核。"""
        block_count = (logical_length + self.block_size - 1) // self.block_size
        block_ids = torch.tensor(
            [self.block_tables[request_id][:block_count] for request_id in request_ids],
            dtype=torch.long,
            device=self.device,
        )
        key = self.blocks[layer_index, 0, block_ids]
        value = self.blocks[layer_index, 1, block_ids]
        # 形状：[批次, 逻辑块数, KV头数, 块内token数, 每头维度]。
        # 调整维度后合并块与块内位置，再裁成 [批次, KV头数, 逻辑长度, 每头维度]。
        key = key.permute(0, 2, 1, 3, 4).reshape(
            len(request_ids), self.config.num_key_value_heads,
            block_count * self.block_size, self.config.head_dim,
        )[:, :, :logical_length, :]
        value = value.permute(0, 2, 1, 3, 4).reshape(
            len(request_ids), self.config.num_key_value_heads,
            block_count * self.block_size, self.config.head_dim,
        )[:, :, :logical_length, :]
        return key, value

    def block_id_tensor(self, request_ids, logical_length):
        """把请求的逻辑块表打包成设备上的整数张量，供同长度请求分组读取使用。"""
        block_count = (logical_length + self.block_size - 1) // self.block_size
        return torch.tensor(
            [self.block_tables[request_id][:block_count] for request_id in request_ids],
            dtype=torch.long,
            device=self.device,
        )

    def read_blocks(self, layer_index, block_ids):
        """按给定物理块 ID 读取某层 K/V，保留块维度，交给分页参考 Attention 处理。"""
        return (
            self.blocks[layer_index, 0, block_ids],
            self.blocks[layer_index, 1, block_ids],
        )

    def release(self, request_id):
        """移除请求的逻辑块表与长度，按本章的所有权规则归还可释放的物理块。

        只释放分配关系，不搬动其他请求的数据；共享前缀版本还要保留被引用的块。"""
        table = self.block_tables.pop(request_id)
        self.sequence_lengths.pop(request_id)
        for block_id in table:
            heapq.heappush(self.free_block_ids, block_id)
        self.release_count += len(table)

    def snapshot(self):
        """返回块池容量、占用与分配统计的可记录快照，供测试和实验报告检查资源变化。"""
        return {
            "block_size": self.block_size,
            "pool_blocks": self.max_blocks,
            "used_blocks": self.used_block_count,
            "free_blocks": len(self.free_block_ids),
            "peak_used_blocks": self.peak_used_blocks,
            "live_tokens": self.live_token_count,
            "allocated_token_slots": self.allocated_token_slots,
            "cache_utilization": self.utilization,
            "live_cache_bytes": self.live_cache_bytes,
            "pool_bytes": self.pool_bytes,
            "allocation_count": self.allocation_count,
            "reuse_count": self.reuse_count,
            "release_count": self.release_count,
            "block_tables": {
                request_id: list(table)
                for request_id, table in self.block_tables.items()
            },
        }


def paged_decode_forward(model, input_ids, position_ids, slot_request_ids, cache):
    """为活跃请求预留本轮 token 的位置，按有效上下文长度分组执行分页参考 Attention。

    逐层写入新 K/V 后，再按块表读取对应历史；尾块未使用位置必须屏蔽。
    所有层完成后统一递增长度，返回 logits 和逻辑访问槽位数。具体后端读取
    方式以本章分支为准；它是教学实现，分页存储不自动意味着融合的分页内核。"""

    if input_ids.shape[1] != 1:
        raise ValueError("Paged Decode 每次只接受一个新 Token")
    active = [
        (slot, request_id)
        for slot, request_id in enumerate(slot_request_ids)
        if request_id is not None
    ]
    if not active:
        raise ValueError("Paged Decode 至少需要一个活跃请求")
    request_ids = [request_id for _, request_id in active]
    append_positions = cache.prepare_append(request_ids)

    groups = defaultdict(list)
    for slot, request_id in active:
        # 本轮新 token 也参与注意力，所以可见长度是旧长度加 1；按该长度把请求分组。
        groups[append_positions[request_id] + 1].append((slot, request_id))
    group_plans = []
    for logical_length, members in groups.items():
        rows = torch.tensor(
            [slot for slot, _ in members], dtype=torch.long, device=input_ids.device
        )
        member_ids = [request_id for _, request_id in members]
        group_plans.append((
            logical_length,
            rows,
            cache.block_id_tensor(member_ids, logical_length),
        ))

    hidden_states = model.model.embed_tokens(input_ids)
    cosine, sine = model.model.rotary_emb(position_ids, hidden_states.dtype)
    visited_token_slots = 0

    for layer_index, layer in enumerate(model.model.layers):
        residual = hidden_states
        normalized = layer.input_layernorm(hidden_states)
        attention = layer.self_attn
        batch_size = normalized.shape[0]
        query = attention.q_norm(
            attention.q_proj(normalized).view(
                batch_size, 1, attention.num_attention_heads,
                attention.head_dim,
            )
        ).transpose(1, 2)
        key = attention.k_norm(
            attention.k_proj(normalized).view(
                batch_size, 1, attention.num_key_value_heads,
                attention.head_dim,
            )
        ).transpose(1, 2)
        value = attention.v_proj(normalized).view(
            batch_size, 1, attention.num_key_value_heads,
            attention.head_dim,
        ).transpose(1, 2)
        query, key = apply_rotary_embedding(query, key, cosine, sine)

        for slot, request_id in active:
            position = append_positions[request_id]
            cache.write_token(
                layer_index, request_id, key[slot], value[slot], position
            )

        attention_output = torch.zeros_like(query)
        for logical_length, rows, block_ids in group_plans:
            group_key, group_value = cache.read_blocks(layer_index, block_ids)
            # 形状：[批次, 逻辑块数, KV头数, 块内token数, 每头维度]。
            group_key = group_key.repeat_interleave(
                attention.num_key_value_groups, dim=2
            )
            group_value = group_value.repeat_interleave(
                attention.num_key_value_groups, dim=2
            )
            group_query = query[rows]
            scores = torch.einsum(
                "bhqd,bnhsd->bhqns", group_query, group_key
            )
            scores = scores / math.sqrt(attention.head_dim)
            tail = logical_length % cache.block_size
            if tail:
                # 尾块空余槽位可能含旧数据，必须在 Softmax 前屏蔽，不能把它们当作额外历史。
                scores[..., -1, tail:] = torch.finfo(scores.dtype).min
            flat_scores = scores.flatten(start_dim=-2)
            probabilities = F.softmax(
                flat_scores, dim=-1, dtype=torch.float32
            ).to(group_query.dtype).view_as(scores)
            attention_output[rows] = torch.einsum(
                "bhqns,bnhsd->bhqd", probabilities, group_value
            )
            if layer_index == 0:
                visited_token_slots += rows.numel() * logical_length

        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, 1, -1
        )
        hidden_states = residual + attention.o_proj(attention_output)
        residual = hidden_states
        hidden_states = residual + layer.mlp(
            layer.post_attention_layernorm(hidden_states)
        )

    # 只有全部模型层都写完，本轮 token 才在每个请求中正式变为有效历史。
    cache.finish_append(request_ids)
    hidden_states = model.model.norm(hidden_states)
    return model.lm_head(hidden_states), visited_token_slots
